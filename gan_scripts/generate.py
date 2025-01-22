import numpy as np
import matplotlib.pyplot as plt
from keras import Model
from keras.api.saving import load_model
from gan_preprocessing import load_data, reverse_ecg_normalization, reverse_crf_to_df, Model_path, scaler

_, m_scaler, c_bin_minmax_scaler = load_data(segment_length=1)
seconds_to_generate = 60
Leads = ['III', 'V3', 'V5']

model: Model = load_model(f"{Model_path}/generator3.keras")
noise = np.random.normal(0, 1, (1, 100))
gen_ecgs, gen_crfs_n, gen_crfs_g, gen_crfs_s, gen_crfs_v = model.predict(noise)

# Concatenate the generated segments
gen_ecgs_full = np.concatenate(gen_ecgs, axis=0)
gen_ecgs_full = reverse_ecg_normalization(gen_ecgs_full, m_scaler)
gen_crfs = [gen_crfs_g[0][0], gen_crfs_n[0][0], gen_crfs_n[0][1], gen_crfs_n[0]
            [2], gen_crfs_s[0][0], gen_crfs_n[0][3], gen_crfs_n[0][4], gen_crfs_v[0][0]]
gen_crfs_full = reverse_crf_to_df(gen_crfs, scaler, c_bin_minmax_scaler, col_names=[
    'Gender', 'Age', 'Weight', 'Height', 'Smoker', 'SBP', 'DBP', 'Vascular Event'])
print(gen_crfs_full)
plt.figure(0, figsize=(15, 10))
for lead_idx in range(gen_ecgs_full.shape[1]):
    plt.subplot(3, 1, lead_idx + 1)
    plt.plot(gen_ecgs_full[:, lead_idx], label=f'Lead {Leads[lead_idx]}')
    plt.title(f'Fake ECG - Lead {Leads[lead_idx]}')

plt.tight_layout()
plt.show()
