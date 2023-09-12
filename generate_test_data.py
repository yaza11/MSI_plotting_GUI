"""
File for generating test data
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""
masses and names generated in compound_objects.py by

ms = []
ns = []
for i in range(4, 10):
    ms.append(menaquinone(i, i).mass)
    ns.append(menaquinone(i, i).name)

ms = []
ns = []
for i in range(4, 10):
    ms.append(ubiquinone(i, i).mass)
    ns.append(ubiquinone(i, i).name)
"""

frac_p1 = [444.30283052192,
           512.36543077848,
           580.42803103504,
           648.4906312916,
           716.55323154816,
           784.61583180472]
t_p1 = ['Menaquinone 4:4',
        'Menaquinone 5:5',
        'Menaquinone 6:6',
        'Menaquinone 7:7',
        'Menaquinone 8:8',
        'Menaquinone 9:9']

frac_p2 = [454.30830982518,
           522.37091008174,
           590.4335103383,
           658.49611059486,
           726.5587108514201,
           794.62131110798]
t_p2 = ['Ubiquinone 4:4',
        'Ubiquinone 5:5',
        'Ubiquinone 6:6',
        'Ubiquinone 7:7',
        'Ubiquinone 8:8',
        'Ubiquinone 9:9']

masses = frac_p1 + frac_p2
titles = t_p1 + t_p2

# write mass files
df = pd.DataFrame({'mass': masses, 'name': titles})
df.to_excel('test_mass_file.xlsx', index=False)
df.to_csv('test_mass_file.txt', index=False)


img_size = (50, 30)  # N rows, N_cols
img_height, img_width = img_size
a = np.zeros((img_size[0], img_size[1], len(masses)))

for i, mass in enumerate(masses):
    # average depth of horizon
    if mass in frac_p1:
        d = img_height / 3 + np.random.random() * 6 - 3
    else:
        d = 2 * img_height / 3 + np.random.random() * 6 - 3
    # width of horizon
    wh = img_height / 5 + np.random.random() * 5 - 2.5
    d_offset = i + np.random.random() * 2 - 1

    l_min = round(d - wh / 2 + d_offset)
    l_max = round(d + wh / 2 + d_offset)

    img = np.zeros(img_size)
    # average intensity
    img[l_min:l_max, :] = np.random.random() * 100
    # add gaussian noise
    img += np.random.normal(loc=10, scale=5.0, size=img_size)
    img[img < 0] = 0
    plt.imshow(img)
    plt.title(titles[i])
    plt.show()
    a[:, :, i] = img

# write in spectral form
masses = np.array(masses, dtype=float)
N_masses = len(masses)
noise_threshold = 5
with open('test_data.txt', 'w') as f:
    f.write(str(img_size[0] * img_size[1]) + '\n')
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            # pixel index
            p = f'R00X{str(i).zfill(3)}Y{str(j).zfill(3)}'
            N_mzs = 60
            # add some noise to the mass values
            _masses = masses + np.random.normal(loc=0, scale=3e-3, size=N_masses)
            # fill with some random masses
            noisy_m = np.random.random(N_mzs - len(_masses)) * 400 + 400
            noisy_I = np.random.normal(loc=.5, scale=1, size=300)
            noisy_I[noisy_I < 0] = 0
            _masses = np.concatenate((_masses, noisy_m))
            _intensities = np.concatenate((a[i, j, :], noisy_I))
            _SNR = _intensities / noise_threshold
            o = np.argsort(_masses)
            # sort
            _masses = _masses[o]
            _intensities = _intensities[o]
            _SNR = _SNR[o]
            v = np.vstack((_masses, _intensities, _SNR)).T.flatten().astype(str)
            line = ';'.join([p, str(N_mzs)] + list(v)) + '\n'
            f.write(line)
