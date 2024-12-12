import os
import pandas as pd
from Bio import SeqIO, AlignIO, Phylo
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio.Align.Applications import MafftCommandline
from collections import defaultdict
import subprocess
import tempfile
import logging

# Set up logging
logging.basicConfig(filename="consensus_generation.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Set up file paths in the current working directory
cwd = os.getcwd()
csv_file = os.path.join(cwd, "Extracted_combining_files_Final.csv")
xlsx_file = os.path.join(cwd, "species_sequences.xlsx")
combined_sequences_file = os.path.join(cwd, "combined_sequences.fasta")
aligned_file = os.path.join(cwd, "aligned_sequences.fasta")
cleaned_aligned_file = os.path.join(cwd, "aligned_sequences_cleaned.fasta")
tree_output = os.path.join(cwd, "phylogenetic_tree.nwk")

# Step 1: Load sequences from CSV with consensus for duplicates
species_sequences = defaultdict(list)
print("Step 1: Loading sequences from CSV file and generating consensus for duplicates...")
try:
    df_csv = pd.read_csv(csv_file)
    sequences_csv = df_csv.iloc[:, 1].values  # Column 2 for sequences
    species_csv = df_csv.iloc[:, 7].values    # Column 8 for species (genus and species full name)

    for seq, sp in zip(sequences_csv, species_csv):
        sp_full = sp.replace(" ", "_")
        species_sequences[sp_full].append(SeqRecord(Seq(seq), id=sp_full, description=""))

    consensus_records = []
    for sp_full, records in species_sequences.items():
        if len(records) > 1:
            try:
                # Generate consensus sequence for duplicates
                with tempfile.NamedTemporaryFile(mode="w+", suffix=".fasta", delete=False) as temp_fasta:
                    SeqIO.write(records, temp_fasta, "fasta")
                    temp_fasta_path = temp_fasta.name

                # Align sequences using MAFFT
                mafft_cline = MafftCommandline(cmd="mafft", input=temp_fasta_path)
                stdout, stderr = mafft_cline()
                with tempfile.NamedTemporaryFile(mode="w+", suffix=".fasta", delete=False) as aligned_temp:
                    aligned_temp.write(stdout)
                    aligned_temp_path = aligned_temp.name

                aligned_sequences = AlignIO.read(aligned_temp_path, "fasta")
                consensus_seq = ""
                for i in range(aligned_sequences.get_alignment_length()):
                    column = aligned_sequences[:, i]
                    consensus_base = max(set(column), key=column.count)  # Most common base
                    consensus_seq += consensus_base

                consensus_record = SeqRecord(Seq(consensus_seq), id=sp_full, description="consensus")
                consensus_records.append(consensus_record)

                os.remove(temp_fasta_path)
                os.remove(aligned_temp_path)
            except Exception as e:
                logging.error(f"Error generating consensus for {sp_full}: {e}")
        else:
            consensus_records.append(records[0])
except Exception as e:
    logging.error(f"Error loading CSV file: {e}")
    exit()

# Step 2: Load sequences from Excel and keep as is
print("Step 2: Loading sequences from Excel file without consensus generation...")
try:
    df_xlsx = pd.read_excel(xlsx_file)
    for index, row in df_xlsx.iterrows():
        sp_full = row.iloc[0].replace(" ", "_")  # First column: species name
        sequence = row.iloc[1]  # Second column: sequence
        consensus_records.append(SeqRecord(Seq(sequence), id=sp_full, description=""))
except Exception as e:
    logging.error(f"Error loading Excel file: {e}")
    exit()

# Step 3: Write combined consensus sequences to a single FASTA file
print("Step 3: Writing combined consensus and original sequences to file...")
try:
    SeqIO.write(consensus_records, combined_sequences_file, "fasta")
    logging.info(f"Combined sequences with consensus written to {combined_sequences_file}")
except Exception as e:
    logging.error(f"Error writing combined sequences: {e}")
    exit()

# Step 4: Perform Multiple Sequence Alignment using MAFFT
print("Step 4: Performing alignment using MAFFT...")
try:
    mafft_cline = MafftCommandline(cmd="mafft", input=combined_sequences_file, thread=16)
    stdout, stderr = mafft_cline()
    with open(aligned_file, "w") as aligned_output:
        aligned_output.write(stdout)
    logging.info(f"MAFFT alignment completed and saved to {aligned_file}")
except Exception as e:
    logging.error(f"MAFFT alignment failed: {e}")
    exit()

# Step 5: Clean taxon names for unique identifiers in the alignment
print("Step 5: Cleaning taxon names...")
try:
    unique_records = []
    taxon_count = defaultdict(int)
    with open(aligned_file, "r") as infile:
        for record in SeqIO.parse(infile, "fasta"):
            clean_id = record.id.replace("[", "").replace("]", "").replace(" ", "_")
            taxon_count[clean_id] += 1
            if taxon_count[clean_id] > 1:
                clean_id = f"{clean_id}_{taxon_count[clean_id]}"
            record.id = clean_id
            record.description = ""
            unique_records.append(record)

    with open(cleaned_aligned_file, "w") as outfile:
        SeqIO.write(unique_records, outfile, "fasta")
    logging.info(f"Cleaned alignment with unique names saved to {cleaned_aligned_file}")
except Exception as e:
    logging.error(f"Error cleaning taxon names: {e}")
    exit()

# Step 6: Build Neighbor-Joining Tree with FastTree
print("Step 6: Building tree with FastTree...")
try:
    fasttree_command = ["fasttree", "-nt", cleaned_aligned_file]
    with open(tree_output, "w") as tree_file:
        subprocess.run(fasttree_command, stdout=tree_file, check=True)
    logging.info(f"Tree saved to {tree_output}")
except subprocess.CalledProcessError as e:
    logging.error(f"FastTree execution failed: {e}")
    exit()

# Step 7: Extract and save node and edge data for machine learning applications
print("Step 7: Extracting node and edge data for GNN...")
try:
    # Load the tree to extract edges and nodes
    tree = Phylo.read(tree_output, "newick")
    edges = []
    nodes = set()
    node_counter = 0  # Counter for unnamed nodes

    # Function to traverse the tree and collect edges
    def extract_edges(clade, parent=None):
        global node_counter
        # Assign a unique name to unnamed internal nodes
        node_name = clade.name if clade.name else f"UnnamedNode_{node_counter}"
        if not clade.name:
            node_counter += 1

        # Add the node to the nodes set
        nodes.add(node_name)
        if parent is not None:
            edges.append((parent, node_name))  # Store edge as (parent, child)
            print(f"Edge added: {parent} -> {node_name}")  # Debugging output

        # Recursively add each child, treating this node as the parent
        for child in clade.clades:
            extract_edges(child, node_name)

    # Start extracting edges from the root
    extract_edges(tree.root)

    # Save edges to a CSV file
    edge_list_path = os.path.join(cwd, "phylogenetic_edges.csv")
    with open(edge_list_path, "w") as f:
        f.write("Node1,Node2\n")  # Header
        for edge in edges:
            f.write(f"{edge[0]},{edge[1]}\n")
    
    # Save nodes to a separate CSV file if needed
    node_list_path = os.path.join(cwd, "phylogenetic_nodes.csv")
    with open(node_list_path, "w") as f:
        f.write("Node\n")  # Header
        for node in nodes:
            f.write(f"{node}\n")

    logging.info(f"Edge list saved to '{edge_list_path}' and nodes to '{node_list_path}'")
except Exception as e:
    logging.error(f"Error extracting edges and nodes: {e}")
    exit()

print("Process completed successfully.")
