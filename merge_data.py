import os
import json
import pandas as pd

verdict_df = pd.read_json('ise-dsc01/output/output_verdict.json', orient='index')
evidenc_df = pd.read_json('ise-dsc01/public_test.json', orient='index')

output_df = pd.merge(verdict_df, evidenc_df, left_index=True, right_index=True)
output_df = output_df[[0, 'evidence_predict']].rename(columns={0: "verdict", "evidence_predict": "evidence"})
output_df.loc[output_df['verdict'] == 'NEI', 'evidence'] = ""

print(output_df)

output_df.to_json(os.path.join('ise-dsc01/output', "public_result.json"),orient="index", force_ascii=False)
