# **Documentation Organization**

Welcome to the "AI From Scratch to Scale" project documentation. This directory contains all the essential documentation organized into logical categories for easy navigation.

## **📁 Directory Structure**

```
docs\
├── README.md                    # This file - documentation guide
├── AI_Development_Guide.md      # 🎯 MAIN ENTRY POINT - Quick reference for AI development
├── strategy\                    # High-level planning and approach documents
│   ├── Project_Charter.md       # Overall project vision, goals, and 25-model roadmap
│   ├── Dataset_Strategy.md      # Dataset selection and experimental design
│   └── Notebook_Strategy.md     # Jupyter notebook implementation approach
├── technical\                   # Implementation details and architecture
│   ├── Codebase_Architecture.md # Technical architecture and shared infrastructure
│   └── Coding_Standards.md      # Development standards and best practices
└── templates\                   # Code templates and skeleton files (coming soon)
    └── (Template files will be added here)
```

## **🎯 Where to Start**

### **For AI Assistants & Developers**
**Start here**: [`AI_Development_Guide.md`](AI_Development_Guide.md)
- Consolidated quick reference for development
- Essential commands and workflows
- Links to detailed documentation

### **For Project Understanding**
**Read these in order**:
1. [`strategy/Project_Charter.md`](strategy/Project_Charter.md) - Project vision and roadmap
2. [`technical/Codebase_Architecture.md`](technical/Codebase_Architecture.md) - Technical architecture
3. [`strategy/Dataset_Strategy.md`](strategy/Dataset_Strategy.md) - Experimental design

## **📚 Document Categories**

### **🎯 Strategy Documents** (`strategy/`)
High-level planning, vision, and approach documents that define **what** we're building and **why**.

- **[Project_Charter.md](strategy/Project_Charter.md)**: The master plan
  - Project goals and philosophy
  - 25-model roadmap across 6 modules
  - Learning methodology and engagement levels
  - Historical fidelity approach

- **[Dataset_Strategy.md](strategy/Dataset_Strategy.md)**: Experimental design
  - Strength/weakness dataset approach
  - Specific datasets for each of the 25 models
  - Educational rationale for dataset choices

- **[Notebook_Strategy.md](strategy/Notebook_Strategy.md)**: Analysis approach
  - Three-notebook model for each implementation
  - Theory → Code → Analysis workflow
  - Jupyter notebook best practices

### **🔧 Technical Documents** (`technical/`)
Implementation details, architecture, and development standards that define **how** we build.

- **[Codebase_Architecture.md](technical/Codebase_Architecture.md)**: System design
  - Shared infrastructure vs. model-specific code
  - Directory structure and organization
  - Training/evaluation/visualization workflows
  - Dependency management strategy

- **[Coding_Standards.md](technical/Coding_Standards.md)**: Development practices
  - Code style and formatting requirements
  - Documentation standards
  - Testing and quality assurance
  - Performance optimization guidelines

### **📄 Templates** (`templates/`)
Code templates, skeleton files, and boilerplate code for consistent implementation.

- **[model.py](templates/model.py)**: Template for neural network model implementation
- **[train.py](templates/train.py)**: Template for training script with argument parsing
- **[config.py](templates/config.py)**: Template for configuration management
- **[constants.py](templates/constants.py)**: Template for model constants and metadata
- **[requirements.txt](templates/requirements.txt)**: Template for dependency management

*Coming soon*:
- **evaluate.py**: Template for model evaluation script
- Notebook templates for the three-notebook approach
- Testing templates and examples

## **🔗 Cross-References**

The documentation is designed to work together:

- **AI_Development_Guide.md** ← Quick reference that links to all other docs
- **Project_Charter.md** ← Defines the "what" and "why"
- **Codebase_Architecture.md** ← Defines the "how" (technical implementation)
- **Dataset_Strategy.md** ← Defines experimental approach for each model
- **Coding_Standards.md** ← Defines quality and style requirements
- **Notebook_Strategy.md** ← Defines analysis and documentation approach

## **📝 Document Maintenance**

### **Adding New Documents**
- **Strategy documents**: Add to `strategy/` for planning and approach
- **Technical documents**: Add to `technical/` for implementation details
- **Templates**: Add to `templates/` for reusable code patterns

### **Updating Links**
When moving or renaming documents, update references in:
- This README.md
- AI_Development_Guide.md (Quick Links section)
- Any cross-references in other documents

## **🎓 Learning Path**

### **For Understanding the Project**
1. [`strategy/Project_Charter.md`](strategy/Project_Charter.md) - Get the big picture
2. [`AI_Development_Guide.md`](AI_Development_Guide.md) - Understand the development approach
3. [`strategy/Dataset_Strategy.md`](strategy/Dataset_Strategy.md) - See the experimental design

### **For Implementation**
1. [`AI_Development_Guide.md`](AI_Development_Guide.md) - Primary reference
2. [`technical/Codebase_Architecture.md`](technical/Codebase_Architecture.md) - Architecture details
3. [`technical/Coding_Standards.md`](technical/Coding_Standards.md) - Quality standards
4. [`strategy/Notebook_Strategy.md`](strategy/Notebook_Strategy.md) - Analysis approach

---

**💡 Tip**: Bookmark the [`AI_Development_Guide.md`](AI_Development_Guide.md) - it's your one-stop reference for active development! 