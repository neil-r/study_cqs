import study_llm.event_extraction_task.dataset_rams as rams
from dotenv import load_dotenv


load_dotenv()  # load environment variables from .env.
# to use huggingface models (loaded from their website) set HF_API_TOKEN in file "".env"

datasets = ["./data/RAMS_1.0c/data/test.jsonlines"]

event_types = [
    "air_or_missile_strikes",
    "killings",
    "invasions",
    # "retreat",
    "stabbings",
    "hangings",
    "self-directed_battles",
    "stealings",
    "hijackings",
    "surrenderings",
    "bombings",
    "biological_attacks",
    "chemical_poison_attacks",
    "death_caused_by_violent_events",
]

keywords = [
    "Violence",
    "Aggression",
    "Conflict",
    "Riot",
    "Clash",
    "Skirmish",
    "Outburst",
    "Brutality",
    "Assault",
    "Bloodshed",
    "Terror",
    "Hostility",
    "crime",
    "war",
    "Civil unrest",
    "Massacre",
    "Annihilation",
    ## **2. Surrounding**
    "Encircle",
    "Blockade",
    "Encampment",
    "Outpost",
    "Confinement",
    "Perimeter",
    "Cordoning",
    "Bordering",
    "Entrapment",
    "Siege tactics",
    "Military ring",
    "Enclosure",
    "Strategic positioning",
    ## **3. Besieging**
    "Siege",
    "Blockade",
    "Encirclement",
    "Entrapment",
    "Starvation tactics",
    "Surrender demand",
    "Breaching",
    "Bombardment",
    "Isolation",
    "Supply cutoff",
    "Trench warfare",
    "Siege weapons",
    ## **4. Attack**
    "Offensive",
    "Assault",
    "Strike",
    "Raid",
    "Ambush",
    "Blitz",
    "Bombing",
    "Firefight",
    "Missile launch",
    "Shelling",
    "Airstrike",
    "Ground assault",
    "Drone strike",
    "Tactical strike",
    "Invasion",
    "Sabotage",
    "Flanking",
    ## **5. Military Operation**
    "Deployment",
    "Counterattack",
    "Maneuver",
    "Reconnaissance",
    "Special forces",
    "Joint task force",
    "Tactical assault",
    "Covert operation",
    "Strategic strike",
    "Occupation",
    "Military buildup",
    "Retaliation",
    "Air raid",
    "Naval blockade",
    "Infantry advance",
    ## **6. Hostile Encounter**
    "Engagement",
    "Confrontation",
    "Skirmish",
    "Clash",
    "Duel",
    "Shootout",
    "Standoff",
    "Hostilities",
    "Direct conflict",
    "Military contact",
    "Combat zone",
    "Crossfire",
    "Sniper attack",
    ## **7. Terrorism**
    "Terrorist attack",
    "Bombing",
    "Suicide bombing",
    "Hostage situation",
    "Radicalism",
    "Extremism",
    "Insurgent attack",
    "Car bomb",
    "IED",
    "Chemical attack",
    "Lone wolf attack",
    "Hijacking",
    "Mass shooting",
    "Biological attack",
    "Cyberterrorism",
    ## **8. Bearing Arms**
    "Armed",
    "Weaponized",
    "Gunman",
    "Firearms",
    "Ammunition",
    "Armed conflict",
    "Carrying arms",
    "Armament",
    "Rifle",
    "Machine gun",
    "Sidearm",
    "Munitions",
    "Grenade",
    "Tactical gear",
    ## **9. Defending**
    "Defense",
    "Shield",
    "Fortify",
    "Barricade",
    "Countermeasure",
    "Retaliation",
    "Counterattack",
    "Guarding",
    "Cover fire",
    "Resistance",
    "Protection",
    "Defense line",
    "Trenches",
    "Antiaircraft",
    ## **10. Killing**
    "Homicide",
    "Murder",
    "Assassination",
    "Execution",
    "Massacre",
    "Casualty",
    "Genocide",
    "Slaughter",
    "Shooting",
    "Sniping",
    "Decapitation",
    "Lynching",
    "Poisoning",
    "Manslaughter",
    "Ethnic cleansing",
    ### ✅ **Additional Cross-Cutting Terms**
    "War",
    "Conflict",
    "Combat",
    "Armed conflict",
    "Military engagement",
    "Political violence",
    "Insurrection",
    "Civil war",
    "Guerrilla warfare",
    "Militant",
    "Paramilitary",
    "Resistance",
    "Espionage",
]

topic_msg = "attacks, such as " + (
    ", ".join(e.lower().replace("_", " ") for e in event_types[:-1])
    + ", or "
    + event_types[-1].lower().replace("_", " ")
)

event_type_prefixes = [
    "conflict.attack.",
    "life.die.deathcausedbyviolentevents",
]


# Prepare the creator(s) of the prompts
p_factories = [
    rams.DefaultEventExistsClassificationTaskPromptFactory(
        topic_msg, event_type_prefixes
    ),
    rams.DefaultExtractEventsPromptFactory(topic_msg, event_type_prefixes),
]

huggingface_models = [
    "meta-llama/Llama-3.2-3B-Instruct",
    # "mistralai/Mistral-7B-Instruct-v0.3",
    # "meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8"
]


def determine_classification(topic: rams.RamsPassage) -> bool:

    for event in topic.events:
        for event_type_prefix in event_type_prefixes:
            if event.type_indicator[0].startswith(event_type_prefix):
                return True

    return False

