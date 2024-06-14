import pandas as pd
import re

def main():

    file_name = "pingpong.out"

    pattern = r"# Benchmarking (.*)"

    table_found = False
    i = 0
    table = []

    with open(file_name, "r") as file:
        for line in file:
            i += 1
            if not table_found:
                match = re.match(pattern, line)

                if match:
                    captured_group = match.group(1)

                    print(f"Parsing output for {captured_group.strip()} benchmark...")

                    i += 1
                    next_line = next(file, None).strip()

                    process_pattern = r"= (\d+)"

                    pattern_match = re.search(process_pattern, next_line)

                    if pattern_match:
                        i += 1
                        next(file, None).strip()
                    else:
                        print("Error: no process number found")
                        break
                    table_found = True
            else:
                line = line.strip()
                if line== "# All processes entering MPI_Finalize":
                    break
                else:
                    if not line == "":
                        table.append(re.split(r'\s+', line))

    df = pd.DataFrame(table[1:], columns=[table[0]])
    
    print(df["Mbytes/sec"])


    if not table_found:
        print(f"Error: data was not in given file: {file_name}") 




if __name__ == '__main__':
    main()
