import numpy as np
import torch
from torch.utils.data import Dataset
from tonic.download_utils import check_integrity, download_url
import os
import re


# https://github.com/imanefjer/LongListOps/blob/main/data/ListOpsDataset.py
def tokenize(expression):
    """Convert expression string to tokens, preserving operators."""
    # Replace parentheses with spaces
    expr = expression.replace('(', ' ').replace(')', ' ')
    
    # Add spaces around brackets that aren't part of operators
    expr = re.sub(r'\[(?!(MIN|MAX|MED|SM))', ' [ ', expr)
    expr = expr.replace(']', ' ] ')
    
    # Split and filter empty strings
    return [token for token in expr.split() if token]

def load_listops_data(file_path, max_rows=None):
    """
    Load ListOps data from TSV file.
    
    Args:
        file_path: Path to the TSV file
        max_rows: Maximum number of rows to load (for testing)
    
    Returns:
        sources: Array of source expressions
        targets: Array of target values (0-9)
    """
    sources = []
    targets = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        next(f)  # Skip header (Source, Target)
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            
            source, target = line.strip().split('\t')
            sources.append(source)
            targets.append(int(target))  # Target is always 0-9
    
    # Convert to numpy arrays
    source_array = np.array(sources, dtype=object)  # Keep expressions as strings
    target_array = np.array(targets, dtype=np.int32)  # Targets are integers
    
    return source_array, target_array


# https://github.com/imanefjer/LongListOps/blob/main/data/ListOpsDataset.py
class ListOps(Dataset):
    """ListOps, shorter variant"""
    dataset_name = "ListOps"

    base_url = "https://github.com/nyu-mll/spinn/raw/refs/heads/listops-release/python/spinn/data/listops/"
    train_file = "train_d20s.tsv"
    train_md5 = "f3141589f80dad7de8ce2b6bcbbd306b"
    test_file = "test_d20s.tsv"
    test_md5 = "a533eeca66c3b8a9ca7ddc47d4e5254b"

    def __init__(
        self,
        save_to,
        train=True
    ):
        # Create vocabulary from operators and digits
        self.vocab = {
            'PAD': 0,  # Padding token
            '[MIN': 1,
            '[MAX': 2,
            '[MED': 3,
            '[SM': 4,
            ']': 5,
            '(': 6,
            ')': 7
        }
        # Add digits 0-9
        for i in range(10):
            self.vocab[str(i)] = i + 8

        self.location_on_system = os.path.join(save_to, "listops")
        self.train = train

        if train:
            self.file = self.train_file
            self.md5 = self.train_md5
        else:
            self.file = self.test_file
            self.md5 = self.test_md5

        if not self._file_present():
            self.download()

        self.X, self.y = load_listops_data(os.path.join(self.location_on_system, self.file))


    def __getitem__(self, idx):
        expr = self.X[idx]
        target = self.y[idx]
        
        # Convert to token IDs without padding or truncating
        token_ids = self.tokenize(expr)
        
        {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.long)
        }

        return frames, target, block_idx

    def __len__(self):
        return len(self.X)
    
    def tokenize(self, expr):
        """Convert expression to token IDs."""
        tokens = tokenize(expr)
        return [self.vocab.get(token, 0) for token in tokens]

    def _file_present(self) -> bool:
        """Check if the file is present on disk and has the correct md5sum.
        """
        return check_integrity(os.path.join(self.location_on_system,
                                            self.file), md5=self.md5)

    def download(self):
        for (f, m) in [(self.file, self.md5)]:
            download_url(
                self.base_url + f, self.location_on_system, 
                filename=f, md5=m
            )
