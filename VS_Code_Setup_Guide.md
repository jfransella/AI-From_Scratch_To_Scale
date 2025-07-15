# VS Code Setup Guide for AI From Scratch to Scale

This guide will help you set up VS Code for optimal development with our AI project, including Flake8 linting, Black formatting, and more.

## 🚀 Quick Setup

### 1. Install Recommended Extensions

When you open the project in VS Code, you'll be prompted to install recommended extensions. Click **"Install All"** or install manually:

**Essential Extensions:**
- **Python** (`ms-python.python`) - Core Python support
- **Flake8** (`ms-python.flake8`) - Linting with Flake8
- **Black Formatter** (`ms-python.black-formatter`) - Code formatting
- **isort** (`ms-python.isort`) - Import sorting
- **Jupyter** (`ms-toolsai.jupyter`) - Notebook support

### 2. Open as Workspace

For best experience, open the **workspace file** instead of just the folder:
```
File → Open Workspace from File → ai-from-scratch-to-scale.code-workspace
```

This provides:
- ✅ Organized folder structure
- ✅ Pre-configured settings
- ✅ Built-in tasks and debugging

### 3. Select Python Interpreter

1. Press `Ctrl+Shift+P` → type "Python: Select Interpreter"
2. Choose: `./models/01_perceptron/.venv/Scripts/python.exe`
3. This ensures VS Code uses our virtual environment

## 🔧 Configuration Overview

### Automatic Features
- **Format on Save** - Code automatically formatted with Black
- **Organize Imports on Save** - isort organizes imports
- **Real-time Linting** - Flake8 shows issues as you type
- **88 Character Line Ruler** - Visual guide for line length

### Linting Configuration
Our Flake8 setup includes:
```
✅ Compatible with Black formatting
✅ 88 character line length
✅ Project-specific ignore rules
✅ Complexity checking (max 10)
✅ Import order validation
```

### Formatting Configuration
Our Black setup:
```
✅ 88 character line length (PEP 8 extended)
✅ Automatic string quote normalization
✅ Compatible with isort
✅ Python 3.8+ target versions
```

## 🎯 Using the Development Tools

### Tasks (Ctrl+Shift+P → "Tasks: Run Task")

1. **Format Code** - Run Black + isort on current model
2. **Lint Code** - Run Flake8 with problem reporting
3. **Quality Check** - Comprehensive check (format + lint + tests)
4. **Run Tests** - Execute pytest with proper configuration
5. **Train Perceptron** - Start training with visualization

### Debugging (F5 or Debug Panel)

1. **Debug Perceptron Training** - Debug training script
2. **Debug Perceptron Evaluation** - Debug evaluation script  
3. **Debug Current File** - Debug any Python file
4. **Debug Tests** - Debug test files

### Command Palette Shortcuts

| Command | Shortcut | Description |
|---------|----------|-------------|
| Format Document | `Shift+Alt+F` | Format current file |
| Organize Imports | `Shift+Alt+O` | Sort imports |
| Go to Definition | `F12` | Jump to definition |
| Find References | `Shift+F12` | Find all references |
| Rename Symbol | `F2` | Rename variable/function |

## 📝 Development Workflow

### 1. Daily Development
```powershell
# Open VS Code workspace
code ai-from-scratch-to-scale.code-workspace

# VS Code will automatically:
✅ Activate virtual environment
✅ Show linting errors in real-time
✅ Format code on save
✅ Organize imports on save
```

### 2. Before Committing
```powershell
# Run comprehensive check
Ctrl+Shift+P → "Tasks: Run Task" → "Quality Check"

# Or use terminal:
.\scripts\check.ps1 models\01_perceptron\src\
```

### 3. Debugging Issues
```powershell
# Set breakpoints and press F5
# Or use Debug Panel → "Debug Perceptron Training"

# For linting issues:
Ctrl+Shift+P → "Tasks: Run Task" → "Lint Code"
```

## 🎨 Editor Features

### Real-time Feedback
- **Red Squiggles** - Syntax errors
- **Yellow Squiggles** - Linting warnings
- **Blue Squiggles** - Style suggestions
- **Problems Panel** - All issues listed

### IntelliSense
- **Auto-completion** for imports and methods
- **Type hints** and documentation
- **Parameter hints** for functions
- **Import suggestions** for missing modules

### Code Navigation
- **File Explorer** organized by component
- **Outline View** shows class/function structure
- **Breadcrumbs** for navigation context
- **Go to Symbol** (`Ctrl+Shift+O`) for quick jumping

## 🔍 Troubleshooting

### Linting Not Working?
```powershell
# Check Python interpreter
Ctrl+Shift+P → "Python: Select Interpreter"
→ Choose: ./models/01_perceptron/.venv/Scripts/python.exe

# Check Flake8 installation
# In terminal:
.venv\Scripts\activate
flake8 --version
```

### Formatting Not Working?
```powershell
# Check Black installation
black --version

# Manual format:
Shift+Alt+F
```

### Virtual Environment Issues?
```powershell
# Reload window
Ctrl+Shift+P → "Developer: Reload Window"

# Re-select interpreter
Ctrl+Shift+P → "Python: Select Interpreter"
```

### Import Issues?
```powershell
# Check PYTHONPATH
# Should include project root

# Reload window after changing PYTHONPATH
Ctrl+Shift+P → "Developer: Reload Window"
```

## 📁 File Organization

Our workspace organizes files logically:
```
🏠 Project Root          # Main configuration files
📊 Data Utils           # Shared data loading
⚙️ Engine               # Training/evaluation engine  
📈 Plotting             # Visualization utilities
🔧 Utils                # General utilities
🧪 Tests                # All test files
🧠 Models               # Model implementations
01️⃣ Perceptron         # First model (active)
📚 Documentation        # Project docs
```

## 🎯 Pro Tips

### Productivity Shortcuts
- `Ctrl+`` - Open integrated terminal
- `Ctrl+Shift+`` - New terminal
- `Ctrl+K, Ctrl+S` - Keyboard shortcuts reference
- `Ctrl+,` - Open settings

### Multi-file Editing
- `Ctrl+D` - Select next occurrence
- `Alt+Click` - Multiple cursors
- `Ctrl+Shift+L` - Select all occurrences

### Search & Replace
- `Ctrl+F` - Find in file
- `Ctrl+H` - Replace in file
- `Ctrl+Shift+F` - Find in workspace
- `Ctrl+Shift+H` - Replace in workspace

## 🆘 Getting Help

### VS Code Resources
- `F1` → "Help: Show All Commands"
- `Ctrl+Shift+P` → "Help: Welcome"
- View → Command Palette for all commands

### Project-Specific Help
- Check `docs/` folder for project documentation
- Run `.\scripts\check.ps1 --help` for script options
- Use Debug Console for runtime debugging

---

**Happy Coding! 🎉**

Your VS Code is now optimized for AI development with automatic formatting, linting, and intelligent code suggestions. 