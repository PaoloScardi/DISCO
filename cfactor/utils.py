import zipfile
from pathlib import Path

def ensure_mathematica_sources(zip_path: Path, target_dir: Path) -> list[Path]:
    """
    Verifica ed estrae i sorgenti Mathematica dal file zip nella cartella target.
    Ritorna la lista dei file estratti (solo .m).
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip non trovato: {zip_path}")
    target_dir.mkdir(parents=True, exist_ok=True)

    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for info in zf.infolist():
            # estrae solo file, ignora directory
            if info.is_dir():
                continue
            name = Path(info.filename)
            # normalizza sotto-cartelle a target_dir
            dest = target_dir / name.name
            # estrai
            with zf.open(info, 'r') as src, open(dest, 'wb') as dst:
                dst.write(src.read())
            if dest.suffix.lower() == '.m':
                extracted.append(dest)
    return extracted