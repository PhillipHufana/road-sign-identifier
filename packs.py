# core/packs.py
import os
import random

SECTORS = ["Forest", "Disaster", "Public Health", "Agriculture"]

DEFAULT_FACTS = {
    "Forest": [
        "Clear imaging helps detect logging roads and newly cleared patches earlier.",
        "Noise in aerial imagery can hide early canopy loss and forest thinning.",
        "Sharper drone footage supports more accurate forest monitoring over time."
    ],
    "Disaster": [
        "Clear images help validate flood levels and damage faster during response.",
        "Low-contrast photos can hide cracks and landslide lines in terrain.",
        "Better visibility improves documentation for relief coordination."
    ],
    "Public Health": [
        "Clear photos help document sanitation issues and mosquito breeding sites.",
        "Good contrast improves visibility of stagnant water and clogged drainage.",
        "Better image quality supports safer, faster inspection reporting."
    ],
    "Agriculture": [
        "Clear images help spot leaf disease patterns and pest damage early.",
        "Low contrast can hide discoloration signs in crops.",
        "Sharper field photos improve monitoring and extension support."
    ]
}

def list_images_in_folder(folder):
    if not os.path.isdir(folder):
        return []
    return [
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]

def pick_random_sector_image(packs_root: str, sector: str):
    sector_dir = os.path.join(packs_root, sector.lower().replace(" ", "_"))
    files = list_images_in_folder(sector_dir)
    if not files:
        return None
    return random.choice(files)

def random_fact(sector: str):
    facts = DEFAULT_FACTS.get(sector, [])
    if not facts:
        return "Clear images improve reporting and reduce wasted effort in documentation."
    return random.choice(facts)
