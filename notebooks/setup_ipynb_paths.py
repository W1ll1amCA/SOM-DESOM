import sys
from pathlib import Path

def setup_project_root()->Path:
    """Ajoute le répertoire racine du projet au sys.path pour permettre l'importation des modules depuis n'importe quel notebook."""

    # Trouve le répertoire racine du projet en remontant les parents jusqu'à trouver un dossier contenant "scripts"
    PROJECT_ROOT = next(
        p for p in Path.cwd().resolve().parents
        if (p / "scripts").is_dir()
    )
    # Ajoute le répertoire racine du projet au sys.path s'il n'y est pas déjà
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    return PROJECT_ROOT