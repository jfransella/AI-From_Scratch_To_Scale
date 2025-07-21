"""
File organization and naming standards for AI From Scratch to Scale project.

This module implements standardized file organization, naming conventions,
and directory structure management following the project strategy specification.
"""

import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass
from datetime import datetime

from utils import get_logger


# Project structure standards
PROJECT_STRUCTURE = {
    'data': {
        'raw': 'Raw unprocessed datasets',
        'processed': 'Processed and cleaned datasets', 
        'cache': {
            'synthetic': 'Cached synthetic datasets',
            'real': 'Cached real datasets',
            'processed': 'Cached processed datasets',
            'models': 'Cached model checkpoints'
        },
        'external': 'External datasets and downloads'
    },
    'models': {
        'checkpoints': 'Model checkpoints and saved states',
        'configs': 'Model configuration files',
        'outputs': 'Model training outputs and results'
    },
    'experiments': {
        'results': 'Experiment results and logs',
        'configs': 'Experiment configurations',
        'artifacts': 'Experiment artifacts and plots'
    },
    'docs': {
        'api': 'API documentation',
        'examples': 'Usage examples',
        'guides': 'User and developer guides'
    }
}

# Naming conventions
NAMING_CONVENTIONS = {
    'dataset_files': {
        'pattern': r'^[a-z][a-z0-9_]*$',
        'description': 'Lowercase with underscores (snake_case)',
        'examples': ['iris_data', 'mnist_subset', 'xor_problem']
    },
    'model_files': {
        'pattern': r'^[A-Z][a-zA-Z0-9]*$',
        'description': 'PascalCase for model classes',
        'examples': ['Perceptron', 'ADALINE', 'MLP']
    },
    'config_files': {
        'pattern': r'^[a-z][a-z0-9_]*\.py$',
        'description': 'Snake_case for configuration files',
        'examples': ['config.py', 'model_config.py', 'experiment_config.py']
    },
    'script_files': {
        'pattern': r'^[a-z][a-z0-9_]*\.py$',
        'description': 'Snake_case for script files',
        'examples': ['train.py', 'evaluate.py', 'visualize.py']
    },
    'output_directories': {
        'pattern': r'^[a-z][a-z0-9_]*$',
        'description': 'Snake_case for output directories',
        'examples': ['training_outputs', 'experiment_results', 'model_checkpoints']
    }
}

# File type classifications
FILE_TYPES = {
    'data_files': {'.npz', '.npy', '.csv', '.json', '.pkl', '.h5', '.hdf5'},
    'model_files': {'.pth', '.pt', '.ckpt', '.pkl'},
    'config_files': {'.py', '.yaml', '.yml', '.json', '.toml'},
    'documentation': {'.md', '.rst', '.txt'},
    'notebooks': {'.ipynb'},
    'scripts': {'.py', '.sh', '.ps1', '.bat'},
    'images': {'.png', '.jpg', '.jpeg', '.svg', '.pdf'},
    'logs': {'.log', '.txt'}
}

# Standard directory templates
DIRECTORY_TEMPLATES = {
    'model_template': {
        'structure': {
            'src': 'Source code',
            'tests': 'Unit tests',
            'notebooks': 'Jupyter notebooks',
            'outputs': 'Training outputs',
            'configs': 'Configuration files'
        },
        'required_files': ['README.md', 'requirements.txt'],
        'optional_files': ['setup.py', '.gitignore']
    },
    'experiment_template': {
        'structure': {
            'configs': 'Experiment configurations',
            'results': 'Experiment results',
            'plots': 'Generated plots and visualizations',
            'logs': 'Experiment logs'
        },
        'required_files': ['experiment_info.json'],
        'optional_files': ['README.md', 'analysis.ipynb']
    }
}


@dataclass
class FileInfo:
    """Information about a file in the project."""
    path: Path
    name: str
    extension: str
    size_bytes: int
    modified_time: datetime
    file_type: str
    naming_compliant: bool
    issues: List[str]


@dataclass
class DirectoryInfo:
    """Information about a directory in the project."""
    path: Path
    name: str
    structure_compliant: bool
    files: List[FileInfo]
    subdirectories: List['DirectoryInfo']
    issues: List[str]
    template_match: Optional[str] = None


class ProjectOrganizer:
    """
    Project file organization and naming standards manager.
    
    This class provides tools to analyze, validate, and reorganize
    project files according to the established standards.
    """
    
    def __init__(self, project_root: Union[str, Path]):
        """
        Initialize project organizer.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.logger = get_logger(__name__)
        
        if not self.project_root.exists():
            raise ValueError(f"Project root does not exist: {project_root}")
    
    def analyze_project_structure(self) -> Dict[str, any]:
        """
        Analyze current project structure against standards.
        
        Returns:
            Analysis results with compliance information
        """
        self.logger.info(f"Analyzing project structure at {self.project_root}")
        
        analysis = {
            'project_root': str(self.project_root),
            'analysis_time': datetime.now().isoformat(),
            'directory_compliance': {},
            'file_compliance': {},
            'naming_violations': [],
            'structure_violations': [],
            'recommendations': [],
            'summary': {}
        }
        
        # Analyze directory structure
        analysis['directory_compliance'] = self._analyze_directory_structure()
        
        # Analyze file naming
        analysis['file_compliance'] = self._analyze_file_naming()
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        # Create summary
        analysis['summary'] = self._create_analysis_summary(analysis)
        
        self.logger.info(f"Analysis completed: {analysis['summary']['total_files']} files analyzed")
        
        return analysis
    
    def _analyze_directory_structure(self) -> Dict[str, any]:
        """Analyze directory structure compliance."""
        structure_analysis = {
            'expected_directories': [],
            'missing_directories': [],
            'unexpected_directories': [],
            'compliant_directories': [],
            'directory_details': {}
        }
        
        # Check for expected top-level directories
        expected_dirs = set(PROJECT_STRUCTURE.keys())
        existing_dirs = {d.name for d in self.project_root.iterdir() if d.is_dir()}
        
        structure_analysis['expected_directories'] = list(expected_dirs)
        structure_analysis['missing_directories'] = list(expected_dirs - existing_dirs)
        structure_analysis['compliant_directories'] = list(expected_dirs & existing_dirs)
        
        # Analyze each directory in detail
        for dir_path in self.project_root.iterdir():
            if dir_path.is_dir():
                dir_info = self._analyze_single_directory(dir_path)
                structure_analysis['directory_details'][dir_path.name] = dir_info
        
        return structure_analysis
    
    def _analyze_single_directory(self, dir_path: Path) -> Dict[str, any]:
        """Analyze a single directory."""
        analysis = {
            'path': str(dir_path),
            'name_compliant': self._check_directory_naming(dir_path.name),
            'structure_compliant': True,
            'file_count': 0,
            'subdirectory_count': 0,
            'issues': [],
            'files': []
        }
        
        try:
            # Count files and subdirectories
            for item in dir_path.iterdir():
                if item.is_file():
                    analysis['file_count'] += 1
                    file_info = self._analyze_file(item)
                    analysis['files'].append(file_info)
                elif item.is_dir():
                    analysis['subdirectory_count'] += 1
            
        except PermissionError:
            analysis['issues'].append("Permission denied")
        except Exception as e:
            analysis['issues'].append(f"Error analyzing directory: {e}")
        
        return analysis
    
    def _analyze_file_naming(self) -> Dict[str, any]:
        """Analyze file naming compliance."""
        naming_analysis = {
            'total_files': 0,
            'compliant_files': 0,
            'violations': [],
            'file_types': {},
            'naming_patterns': {}
        }
        
        # Walk through all files in project
        for file_path in self.project_root.rglob('*'):
            if file_path.is_file():
                naming_analysis['total_files'] += 1
                
                file_info = self._analyze_file(file_path)
                
                if file_info.naming_compliant:
                    naming_analysis['compliant_files'] += 1
                else:
                    naming_analysis['violations'].append({
                        'file': str(file_path),
                        'issues': file_info.issues
                    })
                
                # Track file type statistics
                if file_info.file_type not in naming_analysis['file_types']:
                    naming_analysis['file_types'][file_info.file_type] = 0
                naming_analysis['file_types'][file_info.file_type] += 1
        
        return naming_analysis
    
    def _analyze_file(self, file_path: Path) -> FileInfo:
        """Analyze a single file."""
        try:
            stat = file_path.stat()
            file_info = FileInfo(
                path=file_path,
                name=file_path.name,
                extension=file_path.suffix.lower(),
                size_bytes=stat.st_size,
                modified_time=datetime.fromtimestamp(stat.st_mtime),
                file_type=self._classify_file_type(file_path),
                naming_compliant=True,
                issues=[]
            )
            
            # Check naming compliance
            file_info.naming_compliant, issues = self._check_file_naming(file_path)
            file_info.issues = issues
            
            return file_info
            
        except Exception as e:
            return FileInfo(
                path=file_path,
                name=file_path.name,
                extension=file_path.suffix.lower(),
                size_bytes=0,
                modified_time=datetime.now(),
                file_type='unknown',
                naming_compliant=False,
                issues=[f"Error analyzing file: {e}"]
            )
    
    def _classify_file_type(self, file_path: Path) -> str:
        """Classify file type based on extension and location."""
        extension = file_path.suffix.lower()
        
        for file_type, extensions in FILE_TYPES.items():
            if extension in extensions:
                return file_type
        
        return 'other'
    
    def _check_file_naming(self, file_path: Path) -> Tuple[bool, List[str]]:
        """Check if file follows naming conventions."""
        issues = []
        
        # Skip hidden files and system files
        if file_path.name.startswith('.'):
            return True, []
        
        # Get relative path for context
        try:
            rel_path = file_path.relative_to(self.project_root)
            path_parts = rel_path.parts
        except ValueError:
            return True, []  # File outside project root
        
        # Check based on file type and location
        file_name = file_path.stem
        extension = file_path.suffix.lower()
        
        # Dataset files in data directories
        if 'data' in path_parts:
            if not re.match(NAMING_CONVENTIONS['dataset_files']['pattern'], file_name):
                issues.append(f"Dataset file should use snake_case: {file_name}")
        
        # Model files
        elif extension in {'.py'} and any('model' in part.lower() for part in path_parts):
            # Check if it's a model class file
            if file_name[0].isupper():
                if not re.match(NAMING_CONVENTIONS['model_files']['pattern'], file_name):
                    issues.append(f"Model file should use PascalCase: {file_name}")
            else:
                if not re.match(NAMING_CONVENTIONS['script_files']['pattern'], file_name):
                    issues.append(f"Script file should use snake_case: {file_name}")
        
        # Configuration files
        elif file_name.endswith('config') or 'config' in file_name:
            if not re.match(NAMING_CONVENTIONS['config_files']['pattern'], file_path.name):
                issues.append(f"Config file should use snake_case: {file_path.name}")
        
        # General Python scripts
        elif extension == '.py':
            if not re.match(NAMING_CONVENTIONS['script_files']['pattern'], file_path.name):
                issues.append(f"Python file should use snake_case: {file_path.name}")
        
        return len(issues) == 0, issues
    
    def _check_directory_naming(self, dir_name: str) -> bool:
        """Check if directory follows naming conventions."""
        # Skip hidden directories
        if dir_name.startswith('.'):
            return True
        
        # Check for snake_case pattern
        return re.match(r'^[a-z][a-z0-9_]*$', dir_name) is not None
    
    def _generate_recommendations(self, analysis: Dict[str, any]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Directory structure recommendations
        missing_dirs = analysis['directory_compliance'].get('missing_directories', [])
        if missing_dirs:
            recommendations.append(
                f"Create missing standard directories: {', '.join(missing_dirs)}"
            )
        
        # File naming recommendations
        violations = analysis['file_compliance'].get('violations', [])
        if violations:
            recommendations.append(
                f"Fix {len(violations)} file naming violations (see details)"
            )
        
        # General recommendations
        total_files = analysis['file_compliance'].get('total_files', 0)
        compliant_files = analysis['file_compliance'].get('compliant_files', 0)
        
        if total_files > 0:
            compliance_rate = (compliant_files / total_files) * 100
            if compliance_rate < 90:
                recommendations.append(
                    f"Improve naming compliance from {compliance_rate:.1f}% to 90%+"
                )
        
        return recommendations
    
    def _create_analysis_summary(self, analysis: Dict[str, any]) -> Dict[str, any]:
        """Create analysis summary."""
        file_compliance = analysis.get('file_compliance', {})
        dir_compliance = analysis.get('directory_compliance', {})
        
        total_files = file_compliance.get('total_files', 0)
        compliant_files = file_compliance.get('compliant_files', 0)
        
        return {
            'total_files': total_files,
            'compliant_files': compliant_files,
            'compliance_percentage': (compliant_files / total_files * 100) if total_files > 0 else 100,
            'naming_violations': len(file_compliance.get('violations', [])),
            'missing_directories': len(dir_compliance.get('missing_directories', [])),
            'recommendations_count': len(analysis.get('recommendations', []))
        }
    
    def create_standard_directories(self, dry_run: bool = True) -> Dict[str, List[str]]:
        """
        Create standard project directories.
        
        Args:
            dry_run: If True, only show what would be created
            
        Returns:
            Dictionary with created and skipped directories
        """
        results = {
            'created': [],
            'skipped': [],
            'errors': []
        }
        
        self.logger.info(f"Creating standard directories (dry_run={dry_run})")
        
        def create_directory_structure(base_path: Path, structure: Dict, level: int = 0):
            """Recursively create directory structure."""
            for name, description in structure.items():
                dir_path = base_path / name
                
                if dir_path.exists():
                    results['skipped'].append(str(dir_path))
                    self.logger.debug(f"  {'  ' * level}✓ {dir_path} (exists)")
                else:
                    if not dry_run:
                        try:
                            dir_path.mkdir(parents=True, exist_ok=True)
                            results['created'].append(str(dir_path))
                            self.logger.info(f"  {'  ' * level}+ Created {dir_path}")
                        except Exception as e:
                            results['errors'].append(f"Failed to create {dir_path}: {e}")
                            self.logger.error(f"  {'  ' * level}✗ Failed to create {dir_path}: {e}")
                    else:
                        results['created'].append(str(dir_path))
                        self.logger.info(f"  {'  ' * level}+ Would create {dir_path}")
                
                # Recursively create subdirectories if description is a dict
                if isinstance(description, dict):
                    create_directory_structure(dir_path, description, level + 1)
        
        # Create the standard structure
        create_directory_structure(self.project_root, PROJECT_STRUCTURE)
        
        return results
    
    def suggest_file_renames(self, fix_violations: bool = False) -> Dict[str, List[Dict]]:
        """
        Suggest file renames for naming compliance.
        
        Args:
            fix_violations: If True, actually perform the renames
            
        Returns:
            Dictionary with rename suggestions and actions taken
        """
        results = {
            'suggestions': [],
            'renamed': [],
            'errors': []
        }
        
        self.logger.info(f"Analyzing file naming violations (fix={fix_violations})")
        
        # Get current analysis
        analysis = self.analyze_project_structure()
        violations = analysis['file_compliance'].get('violations', [])
        
        for violation in violations:
            file_path = Path(violation['file'])
            issues = violation['issues']
            
            # Generate suggested name
            suggested_name = self._suggest_corrected_name(file_path)
            
            if suggested_name and suggested_name != file_path.name:
                suggestion = {
                    'original': str(file_path),
                    'suggested': str(file_path.parent / suggested_name),
                    'issues': issues
                }
                results['suggestions'].append(suggestion)
                
                if fix_violations:
                    try:
                        new_path = file_path.parent / suggested_name
                        if not new_path.exists():
                            file_path.rename(new_path)
                            results['renamed'].append(suggestion)
                            self.logger.info(f"Renamed: {file_path.name} → {suggested_name}")
                        else:
                            results['errors'].append(f"Target exists: {new_path}")
                    except Exception as e:
                        results['errors'].append(f"Failed to rename {file_path}: {e}")
        
        return results
    
    def _suggest_corrected_name(self, file_path: Path) -> Optional[str]:
        """Suggest corrected file name."""
        name = file_path.stem
        extension = file_path.suffix
        
        # Convert to snake_case
        # Replace spaces and hyphens with underscores
        corrected = re.sub(r'[-\s]+', '_', name)
        
        # Convert CamelCase to snake_case (except for model files)
        if not self._is_model_file(file_path):
            corrected = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', corrected)
            corrected = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', corrected)
        
        # Convert to lowercase (except for model files)
        if not self._is_model_file(file_path):
            corrected = corrected.lower()
        
        # Remove multiple underscores
        corrected = re.sub(r'_+', '_', corrected)
        
        # Remove leading/trailing underscores
        corrected = corrected.strip('_')
        
        return f"{corrected}{extension}" if corrected else None
    
    def _is_model_file(self, file_path: Path) -> bool:
        """Check if file is a model class file."""
        rel_path = file_path.relative_to(self.project_root)
        return (
            file_path.suffix == '.py' and
            any('model' in part.lower() for part in rel_path.parts) and
            file_path.stem[0].isupper()
        )
    
    def generate_organization_report(self, output_file: Optional[Path] = None) -> str:
        """
        Generate comprehensive organization report.
        
        Args:
            output_file: Optional file to save report
            
        Returns:
            Report content as string
        """
        analysis = self.analyze_project_structure()
        
        report_lines = [
            "# Project Organization Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Project: {self.project_root}",
            "",
            "## Summary",
            f"- Total files: {analysis['summary']['total_files']}",
            f"- Compliant files: {analysis['summary']['compliant_files']}",
            f"- Compliance rate: {analysis['summary']['compliance_percentage']:.1f}%",
            f"- Naming violations: {analysis['summary']['naming_violations']}",
            f"- Missing directories: {analysis['summary']['missing_directories']}",
            "",
        ]
        
        # Directory compliance section
        dir_compliance = analysis['directory_compliance']
        report_lines.extend([
            "## Directory Structure",
            "",
            "### Expected Directories",
            *[f"- {d}" for d in dir_compliance['expected_directories']],
            "",
            "### Missing Directories",
            *[f"- {d}" for d in dir_compliance['missing_directories']],
            "",
        ])
        
        # File compliance section
        violations = analysis['file_compliance'].get('violations', [])
        if violations:
            report_lines.extend([
                "## Naming Violations",
                "",
            ])
            for violation in violations[:10]:  # Show first 10
                report_lines.append(f"**{violation['file']}**")
                for issue in violation['issues']:
                    report_lines.append(f"  - {issue}")
                report_lines.append("")
            
            if len(violations) > 10:
                report_lines.append(f"... and {len(violations) - 10} more violations")
                report_lines.append("")
        
        # Recommendations section
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            report_lines.extend([
                "## Recommendations",
                "",
                *[f"- {rec}" for rec in recommendations],
                ""
            ])
        
        report_content = "\n".join(report_lines)
        
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            self.logger.info(f"Report saved to {output_file}")
        
        return report_content


def analyze_project_organization(project_root: Union[str, Path]) -> Dict[str, any]:
    """
    Convenience function to analyze project organization.
    
    Args:
        project_root: Root directory of the project
        
    Returns:
        Analysis results
    """
    organizer = ProjectOrganizer(project_root)
    return organizer.analyze_project_structure()


def create_project_directories(project_root: Union[str, Path], 
                             dry_run: bool = True) -> Dict[str, List[str]]:
    """
    Convenience function to create standard project directories.
    
    Args:
        project_root: Root directory of the project
        dry_run: If True, only show what would be created
        
    Returns:
        Results of directory creation
    """
    organizer = ProjectOrganizer(project_root)
    return organizer.create_standard_directories(dry_run=dry_run)


def generate_organization_report(project_root: Union[str, Path],
                               output_file: Optional[Path] = None) -> str:
    """
    Convenience function to generate organization report.
    
    Args:
        project_root: Root directory of the project
        output_file: Optional file to save report
        
    Returns:
        Report content
    """
    organizer = ProjectOrganizer(project_root)
    return organizer.generate_organization_report(output_file=output_file) 