from datasets import load_dataset

ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext")

# Ersten Eintrag anschauen
example = ds["train"][0]

# Jeden Key einzeln printen (sonst zu unübersichtlich)
for key, value in example.items():
    print(f"\n=== {key} ===")
    print(value)