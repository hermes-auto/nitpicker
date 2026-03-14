use crate::config::Config;
use crate::debate::{self, DebateMode};
use crate::review::{self, TaskMode};
use eyre::{Result, WrapErr};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, clap::Args)]
pub struct PrArgs {
    /// Full GitHub PR URL (https://github.com/owner/repo/pull/N)
    pub url: Option<String>,
    #[command(flatten)]
    pub common: crate::CommonArgs,
    #[arg(long)]
    pub prompt: Option<String>,
    #[arg(long)]
    pub debate: bool,
    #[arg(long, default_value = "5")]
    pub rounds: usize,
    /// Skip posting review as a PR comment
    #[arg(long)]
    pub no_comment: bool,
}

#[derive(Deserialize)]
struct PrMeta {
    title: String,
    body: String,
    commits: Vec<serde_json::Value>,
}

pub fn check_gh() -> Result<()> {
    let status = Command::new("gh").arg("--version").output();
    match status {
        Ok(o) if o.status.success() => Ok(()),
        _ => eyre::bail!(
            "`gh` CLI not found or not working. Install it from https://cli.github.com/ and run `gh auth login`."
        ),
    }
}

fn fetch_pr_meta(url: Option<&str>, repo: &Path) -> Result<PrMeta> {
    let mut cmd = Command::new("gh");
    cmd.args(["pr", "view", "--json", "title,body,commits"])
        .current_dir(repo);
    if let Some(u) = url {
        cmd.arg(u);
    }
    let out = cmd.output().wrap_err("failed to run gh pr view")?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        eyre::bail!("gh pr view failed: {}", stderr.trim());
    }
    serde_json::from_slice(&out.stdout).wrap_err("failed to parse gh pr view output")
}

pub fn parse_pr_url(url: &str) -> Result<(String, u32)> {
    // expects https://github.com/owner/repo/pull/N
    let url_obj = url::Url::parse(url).wrap_err("invalid URL")?;
    let segments: Vec<&str> = url_obj
        .path_segments()
        .ok_or_else(|| eyre::eyre!("URL has no path"))?
        .filter(|s| !s.is_empty())
        .collect();
    match segments.as_slice() {
        [owner, repo, "pull", n] => {
            let pr_number: u32 = n
                .parse()
                .wrap_err_with(|| format!("PR number `{n}` is not a valid integer"))?;
            Ok((format!("{owner}/{repo}"), pr_number))
        }
        _ => eyre::bail!(
            "expected a URL like https://github.com/owner/repo/pull/N, got: {url}"
        ),
    }
}

fn clone_pr(repo_slug: &str, pr_number: u32, pr_commit_count: usize, dir: &Path) -> Result<()> {
    let clone_url = format!("https://github.com/{repo_slug}.git");
    // fetch enough history so the base branch commit is reachable for diffing
    let depth = std::cmp::max(5, pr_commit_count + 5).to_string();
    let out = Command::new("git")
        .args(["clone", "--depth", &depth, &clone_url, "."])
        .current_dir(dir)
        .output()
        .wrap_err("failed to run git clone")?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        eyre::bail!("git clone failed: {}", stderr.trim());
    }

    let out = Command::new("gh")
        .args(["pr", "checkout", &pr_number.to_string()])
        .current_dir(dir)
        .output()
        .wrap_err("failed to run gh pr checkout")?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        eyre::bail!("gh pr checkout failed: {}", stderr.trim());
    }
    Ok(())
}

fn post_comment(url: Option<&str>, repo: &Path, body: &str) -> Result<()> {
    let mut cmd = Command::new("gh");
    cmd.args(["pr", "comment", "--body", body])
        .current_dir(repo);
    if let Some(u) = url {
        cmd.arg(u);
    }
    let out = cmd.output().wrap_err("failed to run gh pr comment")?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        eyre::bail!("gh pr comment failed: {}", stderr.trim());
    }
    Ok(())
}

fn build_pr_prompt(meta: &PrMeta, diff_context: &str, extra: Option<&str>) -> String {
    let mut parts = vec![format!("## PR: {}", meta.title)];
    if !meta.body.trim().is_empty() {
        parts.push(meta.body.trim().to_string());
    }
    parts.push(diff_context.to_string());
    if let Some(p) = extra {
        if !p.trim().is_empty() {
            parts.push(format!("Additional instructions: {p}"));
        }
    }
    parts.join("\n\n")
}

pub async fn run_pr(args: PrArgs, config: Config) -> Result<()> {
    check_gh()?;

    let verbose = args.common.verbose;

    // for the remote path we hold the TempDir guard here so Drop runs on any exit
    let _tmpdir_guard: Option<tempfile::TempDir>;

    let (repo, url_for_gh): (PathBuf, Option<String>) = if let Some(ref u) = args.url {
        let (repo_slug, pr_number) = parse_pr_url(u)?;

        // fetch metadata first (works from any dir) to get commit count for depth calculation
        let meta_for_depth = fetch_pr_meta(Some(u), Path::new("."))?;
        let pr_commit_count = meta_for_depth.commits.len();

        let tmpdir = tempfile::TempDir::new().wrap_err("failed to create temp dir")?;
        let path = tmpdir.path().to_path_buf();
        clone_pr(&repo_slug, pr_number, pr_commit_count, &path)?;
        _tmpdir_guard = Some(tmpdir);
        (path, Some(u.clone()))
    } else {
        _tmpdir_guard = None;
        let repo = args.common.repo.canonicalize()?;
        if !repo.join(".git").is_dir() {
            eyre::bail!("--repo must point to a git repository (missing .git)");
        }
        (repo, None)
    };

    let meta = fetch_pr_meta(url_for_gh.as_deref(), &repo)?;
    let diff_context = crate::detect_diff_context(&repo)?;
    let full_prompt = build_pr_prompt(&meta, &diff_context, args.prompt.as_deref());

    if args.debate || config.default_debate() {
        if !args.no_comment {
            eprintln!("note: --debate mode output will be printed but not posted as a PR comment (not yet supported)");
        }
        debate::run_debate(
            &repo,
            &full_prompt,
            &config,
            args.rounds,
            verbose,
            DebateMode::Review,
        )
        .await?;
    } else {
        let report = review::run_review(
            &repo,
            &full_prompt,
            &config,
            verbose,
            TaskMode::Review,
        )
        .await?;

        println!("{report}");

        if !args.no_comment {
            let comment = format!(
                "{report}\n\n---\n🔍 Reviewed by [nitpicker](https://github.com/arsenyinfo/nitpicker)"
            );
            post_comment(url_for_gh.as_deref(), &repo, &comment)?;
        }
    }

    // _tmpdir_guard drops here, auto-cleaning the temp dir
    Ok(())
}
