# Markdown Linting Guide

This guide explains the automated markdown linting setup for the AI From Scratch To Scale project.

## Overview

The project uses `markdownlint` to ensure consistent markdown formatting across all 53+ markdown files. The setup
includes:

- **VS Code Integration**: Automatic linting and fixing on save
- **Bulk Scripts**: PowerShell scripts for project-wide operations
- **VS Code Tasks**: Quick access to linting commands
- **Configuration**: Customized rules for documentation projects

## Quick Start

### 1. Install Dependencies

```bash
# Install markdownlint globally (recommended)
npm install -g markdownlint-cli

# Or install as project dependency
npm install
```text`n### 2. VS Code Extension

Install the `markdownlint` extension in VS Code:

- Extension ID: `DavidAnson.vscode-markdownlint`
- Provides real-time linting and auto-fixing

### 3. Test the Setup

```bash
# Check for issues without fixing
npm run check:markdown

# Fix all auto-fixable issues
npm run fix:markdown

# Or use the PowerShell script
.\scripts\fix_markdown.ps1 -CheckOnly
```text`n## Configuration

### `.markdownlint.json`

The main configuration file with rules optimized for documentation:

```json
{
  "default": true,
  "MD013": {
    "line_length": 120,
    "code_blocks": false,
    "tables": false
  },
  "MD024": {
    "siblings_only": true
  },
  "MD025": {
    "front_matter_title": ""
  },
  "MD033": {
    "allowed_elements": ["kbd", "br", "hr"]
  },
  "MD041": false,
  "MD046": {
    "style": "fenced"
  },
  "MD047": false,
  "MD048": {
    "style": "backtick"
  },
  "MD049": {
    "style": "consistent"
  },
  "MD050": {
    "style": "consistent"
  },
  "MD051": false,
  "MD052": false,
  "MD053": false
}
```text`n### Key Rule Explanations

- **MD013**: Line length limit of 120 characters (relaxed for code blocks and tables)
- **MD024**: Allow duplicate headings if they're not siblings
- **MD025**: Allow multiple top-level headings
- **MD033**: Allow specific HTML elements (kbd, br, hr)
- **MD041**: Disable first heading requirement
- **MD046**: Require fenced code blocks
- **MD047**: Disable file ending with single newline
- **MD048**: Require backtick code blocks
- **MD049/MD050**: Consistent emphasis and strong emphasis
- **MD051-053**: Disable link fragment rules

## Automation Options

### 1. VS Code Integration

**Automatic Fixing on Save:**

- Open any `.md` file
- Make changes
- Save the file (Ctrl+S)
- Issues are automatically fixed

**Manual Commands:**

- `Ctrl+Shift+P` → "Markdownlint: Fix all auto-fixable problems"
- `Ctrl+Shift+P` → "Markdownlint: Show all problems"

### 2. VS Code Tasks

Access via `Ctrl+Shift+P` → "Tasks: Run Task":

- **Lint Markdown**: Check for issues without fixing
- **Fix Markdown**: Fix all auto-fixable issues
- **Fix Markdown (Verbose)**: Fix with detailed output

### 3. PowerShell Scripts

**Bulk Operations:**

```powershell
# Check all files for issues
.\scripts\fix_markdown.ps1 -CheckOnly

# Fix all auto-fixable issues (2)
.\scripts\fix_markdown.ps1

# Verbose output
.\scripts\fix_markdown.ps1 -Verbose
```text`n### 4. NPM Scripts

**Package.json Commands:**

```bash
# Check for issues
npm run check:markdown

# Fix issues
npm run fix:markdown

# Lint without fixing
npm run lint:markdown

# Quality check (alias for lint)
npm run quality:check
```text`n## VS Code Settings

The `.vscode/settings.json` includes comprehensive markdown configuration:

### Linting Settings

```json
{
  "markdownlint.enable": true,
  "markdownlint.fixAll": true,
  "markdownlint.run": "onType"
}
```text`n### Editor Settings for Markdown

```json
{
  "[markdown]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll.markdownlint": "explicit"
    },
    "editor.tabSize": 2,
    "editor.insertSpaces": true,
    "editor.wordWrap": "bounded",
    "editor.wordWrapColumn": 120,
    "editor.rulers": [120]
  }
}
```text`n## Ignored Files

The following directories are automatically ignored:

- `node_modules/`
- `.git/`
- `venv/` and `.venv/`
- `__pycache__/`
- `.pytest_cache/`
- `build/` and `dist/`
- `*.egg-info/`
- `.mypy_cache/`
- `test_outputs/` and `outputs/`
- `wandb/` and `mlruns/`

## Common Issues and Solutions

### 1. Line Length Issues (MD013)

**Problem:** Lines longer than 120 characters
**Solution:** Break long lines or add to ignore list

### 2. Heading Issues (MD024)

**Problem:** Duplicate headings
**Solution:** Use different heading levels or add context

### 3. Code Block Issues (MD046)

**Problem:** Inconsistent code block formatting
**Solution:** Use fenced code blocks with backticks

### 4. Emphasis Issues (MD049/MD050)

**Problem:** Inconsistent emphasis formatting
**Solution:** Choose one style (asterisks or underscores) and stick to it

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Markdown Lint
on: [push, pull_request]
jobs:
  markdown-lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - run: npm install
      - run: npm run lint:markdown
```text`n### Pre-commit Hook

The project includes an automated pre-commit hook that runs markdown linting before each commit.

#### How It Works

1. **Automatic Trigger**: Runs before every `git commit`
2. **Staged Files Only**: Only checks markdown files that are staged for commit
3. **Auto-Fix**: Automatically fixes auto-fixable issues and stages them
4. **Commit Prevention**: Prevents commit if non-fixable issues remain

#### Installation

The pre-commit hook is automatically installed when you run:

```powershell
.\scripts\setup_pre_commit.ps1
```text`n#### Manual Installation

If you need to install manually:

1. **Copy the hook file**:

   ```powershell
   Copy-Item "scripts\pre-commit.ps1" ".git\hooks\pre-commit"
   ```text`n2. **Make it executable** (if needed):

   ```powershell
   # On Windows, this is usually not necessary
   ```text`n#### Testing the Hook

1. **Make a change** to any `.md` file
2. **Stage the file**: `git add <filename>`
3. **Try to commit**: `git commit -m "test"`
4. **Watch the hook run** automatically

#### Hook Behavior

**✅ Successful Case**:

```text
🔍 Running pre-commit markdown linting...
📄 Found 1 staged markdown file(s):
  - docs/example.md
🔧 Running markdownlint on staged files...
✅ All markdown files pass linting!
```text`n**❌ Failed Case**:

```text
🔍 Running pre-commit markdown linting...
📄 Found 1 staged markdown file(s):
  - docs/example.md
🔧 Running markdownlint on staged files...
❌ Markdown linting failed. Please fix the following issues:

docs/example.md:5 MD013/line-length Line length [Expected: 120; Actual: 150]

💡 You can fix these issues by:
   1. Running: npm run fix:markdown
   2. Or manually editing the files
   3. Then staging the changes: git add .

🔧 Auto-fixable issues have been automatically fixed.
   Please review and commit the changes.
```text`n#### Bypassing the Hook

If you need to bypass the hook temporarily:

```bash
git commit --no-verify -m "emergency commit"
```text`n**⚠️ Warning**: Only use this for emergency situations. The hook exists to maintain code quality.

## Best Practices

### 1. Regular Maintenance

- Run `npm run check:markdown` before commits
- Use VS Code's automatic fixing on save
- Review linting output regularly

### 2. Rule Customization

- Modify `.markdownlint.json` for project-specific needs
- Document rule changes in this guide
- Consider team preferences when adjusting rules

### 3. Performance

- The PowerShell script efficiently processes all 53+ files
- VS Code integration provides real-time feedback
- Ignore patterns prevent processing unnecessary files

### 4. Documentation

- Keep this guide updated with rule changes
- Document any custom configurations
- Share best practices with team members

## Troubleshooting

### Common Problems

1. **markdownlint not found**

   ```bash
   npm install -g markdownlint-cli
   ```text`n2. **Permission denied on scripts**

   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```text`n3. **VS Code extension not working**
   - Reload VS Code
   - Check extension is installed
   - Verify settings.json configuration

### Getting Help

- Run `npx markdownlint --help` for command options
- Check [markdownlint documentation](https://github.com/DavidAnson/markdownlint)
- Review VS Code extension settings
- Use verbose mode for detailed output: `.\scripts\fix_markdown.ps1 -Verbose`

## File Statistics

- **Total Markdown Files**: 53+
- **Configuration Files**: 3 (`.markdownlint.json`, `package.json`, VS Code settings)
- **Scripts**: 1 PowerShell script
- **VS Code Tasks**: 3 tasks for different operations

This setup provides comprehensive markdown quality assurance for the entire project while maintaining flexibility for
different use cases and team preferences.
