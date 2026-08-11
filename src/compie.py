import os

output_filename = "projekt_code.txt"

with open(output_filename, "w", encoding="utf-8") as outfile:
    for file in sorted(os.listdir(".")):
        if file.endswith(".py") and file != "combine.py":
            outfile.write(f"\n{'='*40}\n")
            outfile.write(f"DATEI: {file}\n")
            outfile.write(f"{'='*40}\n\n")
            try:
                with open(file, "r", encoding="utf-8") as infile:
                    outfile.write(infile.read())
            except Exception as e:
                outfile.write(f"# Fehler beim Lesen: {e}\n")
            outfile.write("\n\n")

print(f"Fertig! Alle Dateien wurden in '{output_filename}' zusammengefasst.")