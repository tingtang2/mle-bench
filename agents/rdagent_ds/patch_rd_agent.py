#!/usr/bin/env python3
"""
Patch RD-Agent's kaggle_crawler.py to skip download when data already exists.
This allows RD-Agent to work with MLE-bench's pre-mounted datasets.
"""
import sys
from pathlib import Path

def patch_kaggle_crawler():
    """Patch kaggle_crawler.py to work without Kaggle API access."""

    rdagent_path = Path("/home/agent/RD-Agent/rdagent/scenarios/kaggle/kaggle_crawler.py")

    if not rdagent_path.exists():
        print(f"Error: {rdagent_path} not found")
        sys.exit(1)

    content = rdagent_path.read_text()

    # Check if already patched
    if "# MLE-bench patch: skip download if data exists" in content:
        print("Already patched")
        return

    # Patch 1: download_data - add early return if data exists
    # Find the function definition and add check right after local_path assignment
    old_start = """def download_data(competition: str, settings: ExtendedBaseSettings, enable_create_debug_data: bool = True) -> None:
    local_path = settings.local_data_path
    if settings.if_using_mle_data:"""

    new_start = """def download_data(competition: str, settings: ExtendedBaseSettings, enable_create_debug_data: bool = True) -> None:
    local_path = settings.local_data_path

    # MLE-bench patch: skip download if data exists
    competition_data_path = Path(local_path) / competition
    if competition_data_path.exists() and list(competition_data_path.iterdir()):
        logger.info(f"Data for {competition} already exists at {competition_data_path}, skipping download")
        # Still create debug data if needed
        if enable_create_debug_data and not Path(f"{local_path}/sample/{competition}").exists():
            create_debug_data(competition, dataset_path=local_path)
        return

    if settings.if_using_mle_data:"""

    if old_start in content:
        content = content.replace(old_start, new_start, 1)
        print(f"Patched download_data to skip if data exists")
    else:
        print(f"Could not find download_data pattern to patch")
        print(f"Searching for alternative pattern...")
        # Try alternative pattern
        alt_pattern = "def download_data(competition: str, settings: ExtendedBaseSettings, enable_create_debug_data: bool = True) -> None:\n    local_path = settings.local_data_path"
        if alt_pattern in content:
            new_pattern = """def download_data(competition: str, settings: ExtendedBaseSettings, enable_create_debug_data: bool = True) -> None:
    local_path = settings.local_data_path

    # MLE-bench patch: skip download if data exists
    competition_data_path = Path(local_path) / competition
    if competition_data_path.exists() and list(competition_data_path.iterdir()):
        logger.info(f"Data for {competition} already exists at {competition_data_path}, skipping download")
        if enable_create_debug_data and not Path(f"{local_path}/sample/{competition}").exists():
            create_debug_data(competition, dataset_path=local_path)
        return
    """
            content = content.replace(alt_pattern, new_pattern, 1)
            print(f"Patched download_data using alternative pattern")
        else:
            print(f"ERROR: Could not patch download_data")
            sys.exit(1)

    # Patch 2: leaderboard_scores - return empty list if no Kaggle API
    old_def2 = '''@cache_with_pickle(hash_func=lambda x: x, force=True)
def leaderboard_scores(competition: str) -> list[float]:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    return [i.score for i in api.competition_leaderboard_view(competition)]'''

    new_def2 = '''@cache_with_pickle(hash_func=lambda x: x, force=True)
def leaderboard_scores(competition: str) -> list[float]:
    # MLE-bench patch: handle missing Kaggle API credentials
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        return [i.score for i in api.competition_leaderboard_view(competition)]
    except (OSError, IOError, Exception) as e:
        logger.warning(f"Cannot fetch leaderboard for {competition}: {e}")
        return []'''

    if old_def2 in content:
        content = content.replace(old_def2, new_def2, 1)
        print(f"Patched leaderboard_scores")
    else:
        print(f"Could not find leaderboard_scores to patch (may already be patched)")

    # Patch 3: get_metric_direction - default to True if no leaderboard
    old_def3 = '''def get_metric_direction(competition: str) -> bool:
    """
    Return **True** if the metric is *bigger is better*, **False** if *smaller is better*.
    """
    if competition == "aerial-cactus-identification":
        return True
    if competition == "leaf-classification":
        return False
    leaderboard = leaderboard_scores(competition)

    return float(leaderboard[0]) > float(leaderboard[-1])'''

    new_def3 = '''def get_metric_direction(competition: str) -> bool:
    """
    Return **True** if the metric is *bigger is better*, **False** if *smaller is better*.
    """
    if competition == "aerial-cactus-identification":
        return True
    if competition == "leaf-classification":
        return False

    # MLE-bench patch: handle empty leaderboard
    leaderboard = leaderboard_scores(competition)
    if not leaderboard or len(leaderboard) < 2:
        logger.warning(f"No leaderboard data for {competition}, defaulting to 'higher is better'")
        return True

    return float(leaderboard[0]) > float(leaderboard[-1])'''

    if old_def3 in content:
        content = content.replace(old_def3, new_def3, 1)
        print(f"Patched get_metric_direction")
    else:
        print(f"Could not find get_metric_direction to patch (may already be patched)")

    rdagent_path.write_text(content)
    print(f"Successfully patched {rdagent_path}")

if __name__ == "__main__":
    patch_kaggle_crawler()
