import os
import pandas as pd
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def coalesce_columns(df: pd.DataFrame, pattern: str):
    """Return a single Series by forward filling across all columns that match *pattern*."""
    cols = [c for c in df.columns if re.match(pattern, c)]
    if not cols:
        return pd.Series([pd.NA] * len(df))
    return df[cols].bfill(axis=1).iloc[:, 0]

def pick_condition(row):
    if 'CellNumber' in row and pd.notna(row['CellNumber']):
        return row['CellNumber']
    if 'BlockList.Sample' in row and pd.notna(row['BlockList.Sample']):
        return row['BlockList.Sample']
    return pd.NA

def process_subject_file(path: str):
    try:
        df_raw = pd.read_csv(path, low_memory=False, header=None)
    except Exception as e:
        logging.error(f"Could not read {path}: {e}")
        return None

    header_rows = df_raw[df_raw.iloc[:, 0] == 'ExperimentName'].index.tolist()
    if not header_rows:
        logging.warning(f"No header markers found in {path}")
        return None

    out_frames = []
    for idx, start in enumerate(header_rows):
        end = header_rows[idx + 1] if idx + 1 < len(header_rows) else len(df_raw)
        block_df = df_raw.iloc[start:end].copy()
        block_df.columns = block_df.iloc[0]  # first row as header
        block_df = block_df.iloc[1:].reset_index(drop=True)

        required_cols = {'Block', 'Trial', 'Procedure[Block]', 'Subject'}
        if not required_cols.issubset(block_df.columns):
            logging.warning(f"Block starting at row {start} missing required columns; skipped")
            continue

        tidy = pd.DataFrame()
        tidy['SubjectID'] = pd.to_numeric(block_df['Subject'], errors='coerce')
        tidy['Block'] = pd.to_numeric(block_df['Block'], errors='coerce')
        tidy['Trial'] = pd.to_numeric(block_df['Trial'], errors='coerce')
        tidy['Procedure'] = block_df['Procedure[Block]']
        tidy['Condition'] = block_df.apply(pick_condition, axis=1)

        tidy['Target.ACC'] = pd.to_numeric(coalesce_columns(block_df, r'Target\d*\.ACC'), errors='coerce')
        tidy['Target.RT'] = pd.to_numeric(coalesce_columns(block_df, r'Target\d*\.RT'), errors='coerce')

        # remove practice rows - COMMENTED OUT TO PRESERVE ALIGNMENT WITH HAPPE INDICES
        # tidy = tidy[~tidy['Procedure'].str.lower().str.startswith('practice', na=False)]

        out_frames.append(tidy)

    if not out_frames:
        return None

    final_df = pd.concat(out_frames, ignore_index=True)
    final_df.dropna(subset=['Block', 'Trial'], inplace=True)
    
    # Add a cumulative trial counter for unambiguous alignment with HAPPE
    final_df['CumulativeTrial'] = range(1, len(final_df) + 1)
    
    final_df = final_df[['SubjectID', 'Block', 'Trial', 'CumulativeTrial', 'Procedure', 
                         'Condition', 'Target.ACC', 'Target.RT']]
    return final_df

def main():
    """
    Main function to run the data cleaning process.
    It finds all subject CSV files, processes them, and saves the cleaned
    "lite" versions to the output directory.
    """
    input_dir = 'behavior-files-fix'
    output_dir = os.path.join(input_dir, 'output')

    # Clear old lite files
    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            if f.endswith('_lite.csv'):
                os.remove(os.path.join(output_dir, f))
    else:
        os.makedirs(output_dir)

    for fname in os.listdir(input_dir):
        if fname.startswith('Subject') and fname.endswith('.csv'):
            logging.info(f"Processing {fname} ...")
            cleaned = process_subject_file(os.path.join(input_dir, fname))
            if cleaned is None:
                logging.warning(f"{fname}: no data extracted")
                continue
            # overwrite SubjectID with id from filename
            sid_match = re.findall(r'(\d+)', fname)
            if sid_match:
                filename_id = int(sid_match[0])
                cleaned['SubjectID'] = filename_id
            # ensure column order
            cleaned = cleaned[['SubjectID', 'Block', 'Trial', 'CumulativeTrial', 'Procedure', 'Condition', 'Target.ACC', 'Target.RT']]
            out_name = fname.replace('.csv', '_lite.csv')
            cleaned.to_csv(os.path.join(output_dir, out_name), index=False)
            logging.info(f"Saved {out_name}")

if __name__ == '__main__':
    main() 