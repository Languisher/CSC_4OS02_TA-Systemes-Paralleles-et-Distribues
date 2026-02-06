# Rapport - TD1

Nan LIN

## 1. Contexte et méthodologie

Ce rapport couvre le **TP1** à partir du code présent dans `tp1/sources/`.

Commandes de compilation utilisées :

```bash
cd tp1/sources
clang++ -std=c++14 -O2 -march=native -Wall Matrix.cpp ProdMatMat.cpp TestProductMatrix.cpp -o TestProductMatrix.exe -lpthread
clang++ -std=c++14 -O2 -march=native -Wall Matrix.cpp test_product_matrice_blas.cpp -o test_product_matrice_blas.exe -lpthread -lblas
mpic++ -std=c++14 -O2 -march=native -Wall calcul_pi.cpp -o calcul_pi.exe
mpic++ -std=c++14 -O2 -march=native -Wall jeton_anneau.cpp -o jeton_anneau.exe
mpic++ -std=c++14 -O2 -march=native -Wall diffusion_hypercube.cpp -o diffusion_hypercube.exe
```

Remarque importante :
- Les programmes MPI ont bien ete executes (`mpiexec`).
- Pour `TestProductMatrix.exe`, la toolchain utilisee ici n'active pas effectivement OpenMP, donc les tests "threads=1/2/4/8" ne montrent pas de vrai gain parallele.

## 2. Produit matrice-matrice

### 2.1 Effet de la taille de la matrice (algo `naive`)

Mesures (`./TestProductMatrix.exe <n> naive 64 1`) :

| n | Temps (s) | MFlops |
|---|---:|---:|
| 512  | 0.733424 | 366.003 |
| 1023 | 0.990246 | 2162.29 |
| 1024 | 8.469070 | 253.568 |
| 1025 | 1.101200 | 1955.85 |
| 2048 | 50.328400 | 341.356 |

Observation : la chute de performance a `n=1024` (puissance de 2) est nette par rapport a `1023/1025`.

Interpretation : en stockage colonne majeure (`i + j*nbRows`), certains ordres de boucles creent des strides regulieres qui accentuent les conflits cache/TLB sur des dimensions puissances de 2.

### 2.2 Permutation des boucles

Les six permutations sont implementees : `ijk`, `jik`, `ikj`, `kij`, `jki`, `kji`.

Mesures (`n=1024`, `./TestProductMatrix.exe 1024 <ordre> 64 1`) :

| ordre | Temps (s) | MFlops |
|---|---:|---:|
| ijk | 1.656020 | 1296.78 |
| jik | 1.657620 | 1295.52 |
| ikj | 8.685590 | 247.247 |
| kij | 8.134560 | 263.995 |
| jki | 0.140721 | 15260.6 |
| kji | 0.183909 | 11676.9 |

Conclusion : `jki` est la meilleure permutation mesuree sur cette machine, puis `kji`.

### 2.3 OpenMP (produit scalaire `parallel_naive`)

Mesures (`n=1024`, `./TestProductMatrix.exe 1024 parallel_naive 16 <threads>`) :

| threads | Temps (s) | MFlops | Speedup vs 1 thread |
|---:|---:|---:|---:|
| 1 | 8.259710 | 259.995 | 1.000 |
| 2 | 8.305410 | 258.564 | 0.995 |
| 4 | 8.314930 | 258.268 | 0.993 |
| 8 | 8.329230 | 257.825 | 0.991 |

Commentaire : pas d'acceleration visible dans cet environnement (OpenMP non effectif pour ce binaire).

### 2.4 Produit par blocs

Mesures (`n=1024`, `./TestProductMatrix.exe 1024 block <blockSize> 1`) :

| blockSize | Temps (s) | MFlops |
|---:|---:|---:|
| 16 | 4.675000 | 459.355 |
| 32 | 7.576710 | 283.432 |
| 64 | 7.177160 | 299.211 |
| 128 | 8.214480 | 261.426 |
| 256 | 8.684400 | 247.281 |
| 512 | 8.667200 | 247.771 |
| 1024 | 8.435340 | 254.582 |

Le meilleur bloc mesure est `blockSize=16`.

### 2.5 Bloc + OpenMP

#### `parallel_block1` (n=1024, blockSize=16)

| threads | Temps (s) | MFlops | Speedup vs 1 thread |
|---:|---:|---:|---:|
| 1 | 4.666840 | 460.158 | 1.000 |
| 2 | 4.687170 | 458.162 | 0.996 |
| 4 | 4.657930 | 461.038 | 1.002 |
| 8 | 4.669730 | 459.873 | 0.999 |

#### `parallel_block2` (n=1024, blockSize=16)

| threads | Temps (s) | MFlops | Speedup vs 1 thread |
|---:|---:|---:|---:|
| 1 | 4.568340 | 470.079 | 1.000 |
| 2 | 4.552330 | 471.733 | 1.004 |
| 4 | 4.507880 | 476.385 | 1.013 |
| 8 | 4.537390 | 473.286 | 1.007 |

Commentaire : les valeurs restent quasi constantes pour les memes raisons (OpenMP non effectif sur ce binaire).

### 2.6 Comparaison avec BLAS

Mesures BLAS (`./test_product_matrice_blas.exe <n>`) :

| n | Temps BLAS (s) | MFlops BLAS |
|---|---:|---:|
| 512  | 0.000567 | 473431 |
| 1023 | 0.004386 | 488189 |
| 1024 | 0.004742 | 452865 |
| 1025 | 0.004014 | 536567 |
| 2048 | 0.031188 | 550849 |

Comparaison BLAS vs `naive` (meme campagne de mesure) :

| n | Temps naive (s) | Temps BLAS (s) | Gain BLAS (naive/BLAS) |
|---|---:|---:|---:|
| 512  | 0.733424 | 0.000567 | x1293.52 |
| 1023 | 0.990246 | 0.004386 | x225.82 |
| 1024 | 8.469070 | 0.004742 | x1785.97 |
| 1025 | 1.101200 | 0.004014 | x274.34 |
| 2048 | 50.328400 | 0.031188 | x1613.71 |

Conclusion : BLAS est tres largement superieure, avec des gains de plusieurs ordres de grandeur selon la taille.

## 3. Partie MPI (jeton anneau, pi stochastique, hypercube)

### 3.1 Jeton en anneau

Commande :

```bash
mpiexec -n 8 ./jeton_anneau.exe
```

Resultat : `Final token value at rank 0: 8`.

### 3.2 Diffusion hypercube

Commande :

```bash
mpiexec -n 8 ./diffusion_hypercube.exe 3
```

Resultat : tous les rangs (0..7) recoivent `token 1234`.

### 3.3 Monte-Carlo de pi en C/MPI

Commande :

```bash
mpiexec -n 8 ./calcul_pi.exe 4000000
```

Resultats :

| Version | Temps (s) | Pi approx |
|---|---:|---:|
| Sequentielle (rank 0) | 0.0766141 | 3.14235 |
| MPI point-a-point | 0.0129550 | 3.14096 |
| MPI reduce | 0.0131600 | 3.14178 |

Speedup (vs sequentiel) :
- MPI point-a-point : `0.0766141 / 0.0129550 = 5.91`
- MPI reduce : `0.0766141 / 0.0131600 = 5.82`

### 3.4 Monte-Carlo de pi en Python (`uv` + `mpi4py`)

Commandes :

```bash
uv run python compute_pi.py --samples 4000000
mpiexec -n 8 uv run python compute_pi.py --samples 4000000 --mpi
```

Resultats :

| Version | Temps (s) | Pi approx |
|---|---:|---:|
| Python sequentiel | 0.047265 | 3.141202000000 |
| Python MPI (8 procs) | 0.987507 | 3.141307000000 |

Observation : sur cette execution, la version Python MPI est plus lente (surcout de lancement/synchronisation important pour cette taille d'echantillon).

## 4. Synthese

- Les implementations demandees dans le sujet sont presentes dans `tp1/sources/`.
- La permutation des boucles impacte fortement les performances ; `jki` est la meilleure dans cette campagne.
- Le produit par blocs ameliore le cas defavorable (`ikj`) ; meilleur bloc mesure : `16`.
- BLAS reste de tres loin la reference de performance.
- En MPI C, le calcul de pi montre un speedup d'environ `x5.8` avec 8 processus.
- Les mesures OpenMP de `TestProductMatrix.exe` doivent etre refaites sur une toolchain OpenMP pleinement active pour observer une acceleration multithread representative.
