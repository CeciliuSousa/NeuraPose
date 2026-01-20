
import os
import sys
import urllib.request
import bz2
from pathlib import Path

def install_openh264():
    # URL oficial Cisco para OpenH264 1.8.0 (versão compatível com OpenCV Windows default)
    URL = "http://ciscobinary.openh264.org/openh264-1.8.0-win64.dll.bz2"
    FILENAME = "openh264-1.8.0-win64.dll.bz2"
    DLL_NAME = "openh264-1.8.0-win64.dll"

    dest_dir = Path.cwd()
    dll_path = dest_dir / DLL_NAME
    archive_path = dest_dir / FILENAME

    print(f"Baixando OpenH264 de: {URL}")
    try:
        urllib.request.urlretrieve(URL, archive_path)
    except Exception as e:
        print(f"Erro ao baixar: {e}")
        return

    print("Extraindo DLL...")
    try:
        with bz2.open(archive_path, "rb") as source, open(dll_path, "wb") as dest:
            data = source.read()
            dest.write(data)
        print(f"DLL extraída com sucesso: {dll_path}")
    except Exception as e:
        print(f"Erro ao extrair: {e}")
        return
    finally:
        if archive_path.exists():
            os.remove(archive_path)

    # Copia para pasta do backend se estivermos na raiz
    backend_dir = dest_dir / "neurapose_backend"
    if backend_dir.exists():
        import shutil
        dest_backend = backend_dir / DLL_NAME
        shutil.copy2(dll_path, dest_backend)
        print(f"Copiado para: {dest_backend}")

    print("\n[SUCESSO] Codec H.264 instalado!")
    print("Agora o OpenCV conseguirá gerar vídeos compatíveis com o navegador.")

if __name__ == "__main__":
    install_openh264()
