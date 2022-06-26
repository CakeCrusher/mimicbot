import typer
from pathlib import Path
from mimicbot import config

def delete_folder(pth) :
  for sub in pth.iterdir() :
      if sub.is_dir() :
          delete_folder(sub)
      else :
          sub.unlink()
  pth.rmdir()

def ensure_app_path(app_path: Path) -> Path:
  while not app_path.exists():
    typer.secho(f"Path [{app_path}] does not exist.", fg=typer.colors.RED)
    app_path: str = typer.prompt("Path to mimicbot data", default=str(config.APP_DIR_PATH))
    app_path = Path(app_path)
    
  return app_path

def app_path_verifier(app_path_str: str) -> None:
  # callback is called twice for some reason
  if not type(app_path_str) == str:
    return config.APP_DIR_PATH
  app_path = Path(app_path_str)
  if app_path.exists():
    typer.confirm(
      typer.style(f"[{app_path_str}] already exists. Do you want to overwrite it?", fg=typer.colors.YELLOW),
      False,
      abort=True,
    )
    delete_folder(app_path)
  return app_path_str