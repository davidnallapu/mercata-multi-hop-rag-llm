import subprocess

def clear_docker_volume(volume_name="weaviate_data"):
    subprocess.run(["docker", "volume", "rm", volume_name], check=True)
    print(f"Deleted volume: {volume_name}")

clear_docker_volume()
