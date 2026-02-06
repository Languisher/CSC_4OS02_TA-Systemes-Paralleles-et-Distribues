# Rapport TP2 - Parallélisation MPI

## 0. Organisation

Les exigences du sujet de `tp2/Readme.md` ont ete implementees dans :
- `tp2/mandelbrot.py` pour les 3 strategies de distribution des lignes.
- `tp2/matvec.py` pour les 2 decoupages du produit matrice-vecteur.

Le code est en Python + `mpi4py`.

## 1. Ensemble de Mandelbrot

### 1.1 Partition statique par blocs de lignes

Strategie implementee (`--mode block`) :
- Repartition equilibree des lignes par blocs contigus.
- Chaque processus calcule ses lignes locales.
- Le rang 0 reconstruit l'image complete et la sauvegarde.

Formules de decoupage :
- `base = H // nbp`
- `reste = H % nbp`
- `Nloc(rank) = base + 1` si `rank < reste`, sinon `base`

### 1.2 Repartition statique amelioree

Strategie implementee (`--mode static`) :
- Les lignes sont ordonnees en alternant haut/bas : `0, H-1, 1, H-2, ...`.
- Distribution round-robin de cette sequence entre processus.

Interet : les zones centrales de Mandelbrot sont en general plus couteuses. Cette distribution melange mieux lignes faciles et difficiles par processus, ce qui reduit le desequilibre par rapport aux blocs contigus.

Limite : la strategie reste statique. Si la charge reelle varie differemment (autres fenetres, autres parametres), il peut persister un desequilibre.

### 1.3 Strategie maitre-esclave

Strategie implementee (`--mode master`) :
- Rang 0 = maitre : envoie dynamiquement une ligne a calculer.
- Rangs 1..nbp-1 = esclaves : calculent puis renvoient la ligne.
- Quand il n'y a plus de travail, le maitre envoie un message d'arret.

Conclusion attendue :
- En general meilleur equilibrage de charge.
- Surcout de communication plus eleve.
- Souvent gagnant quand la variabilite du cout de calcul par ligne est forte.

### 1.4 Campagne de mesure et resultats

Commandes executees :

```bash
mpiexec -n 1 uv run --no-sync python tp2/mandelbrot.py --mode block  --width 768 --height 768 --iter 200
mpiexec -n 2 uv run --no-sync python tp2/mandelbrot.py --mode block  --width 768 --height 768 --iter 200
mpiexec -n 4 uv run --no-sync python tp2/mandelbrot.py --mode block  --width 768 --height 768 --iter 200
mpiexec -n 8 uv run --no-sync python tp2/mandelbrot.py --mode block  --width 768 --height 768 --iter 200
mpiexec -n 1 uv run --no-sync python tp2/mandelbrot.py --mode static --width 768 --height 768 --iter 200
mpiexec -n 2 uv run --no-sync python tp2/mandelbrot.py --mode static --width 768 --height 768 --iter 200
mpiexec -n 4 uv run --no-sync python tp2/mandelbrot.py --mode static --width 768 --height 768 --iter 200
mpiexec -n 8 uv run --no-sync python tp2/mandelbrot.py --mode static --width 768 --height 768 --iter 200
mpiexec -n 1 uv run --no-sync python tp2/mandelbrot.py --mode master --width 768 --height 768 --iter 200
mpiexec -n 2 uv run --no-sync python tp2/mandelbrot.py --mode master --width 768 --height 768 --iter 200
mpiexec -n 4 uv run --no-sync python tp2/mandelbrot.py --mode master --width 768 --height 768 --iter 200
mpiexec -n 8 uv run --no-sync python tp2/mandelbrot.py --mode master --width 768 --height 768 --iter 200
```

Calcul du speedup :
- `S(nbp) = T(1) / T(nbp)` avec `T(nbp) = compute_time` affiche par le programme.

Resultats mesures :

| Strategie | T(1) | T(2) | T(4) | T(8) | S(2) | S(4) | S(8) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bloc | 0.491670 | 0.251415 | 0.137940 | 0.103522 | 1.956 | 3.564 | 4.749 |
| Statique amelioree | 0.488796 | 0.290075 | 0.128920 | 0.084935 | 1.685 | 3.791 | 5.755 |
| Maitre-esclave | 0.485725 | 0.501413 | 0.172789 | 0.092141 | 0.969 | 2.811 | 5.272 |

Interpretation :
- A 8 processus, la meilleure acceleration est obtenue par la repartition statique amelioree.
- Le mode maitre-esclave est penalise a 2 processus par le surcout de communication/ordonnancement, mais devient competitif a 8 processus.
- Le mode bloc reste robuste mais un peu moins performant sur ce cas.

## 2. Produit matrice-vecteur

On traite la matrice definie par :
- `A_ij = ((i + j) mod N) + 1`

### 2.1 Decoupage par colonnes

Strategie implementee (`--mode cols`) :
- Chaque processus possede `Nloc = N / nbp` colonnes.
- Il calcule une contribution partielle de taille `N`.
- Reduction globale par somme (`Allreduce`) pour obtenir le vecteur final sur tous les processus.

### 2.2 Decoupage par lignes

Strategie implementee (`--mode rows`) :
- Chaque processus possede `Nloc = N / nbp` lignes.
- Il calcule directement `Nloc` composantes du resultat.
- Assemblage global du resultat via `Allgatherv`.

### 2.3 Campagne de mesure et resultats

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
mpiexec -n 1 uv run --no-sync python tp2/matvec.py --mode cols --dim 9600 --repeat 5 --check --speedup
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
mpiexec -n 2 uv run --no-sync python tp2/matvec.py --mode cols --dim 9600 --repeat 5 --check --speedup
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
mpiexec -n 4 uv run --no-sync python tp2/matvec.py --mode cols --dim 9600 --repeat 5 --check --speedup
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
mpiexec -n 8 uv run --no-sync python tp2/matvec.py --mode cols --dim 9600 --repeat 5 --check --speedup
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
mpiexec -n 1 uv run --no-sync python tp2/matvec.py --mode rows --dim 9600 --repeat 5 --check --speedup
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
mpiexec -n 2 uv run --no-sync python tp2/matvec.py --mode rows --dim 9600 --repeat 5 --check --speedup
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
mpiexec -n 4 uv run --no-sync python tp2/matvec.py --mode rows --dim 9600 --repeat 5 --check --speedup
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
mpiexec -n 8 uv run --no-sync python tp2/matvec.py --mode rows --dim 9600 --repeat 5 --check --speedup
```

La sortie fournit :
- `nloc`
- `avg_time`
- `speedup` (si `--speedup`)
- verification numerique (`check=OK`).

Resultats mesures :

| Strategie | T(1) | T(2) | T(4) | T(8) | S(2) | S(4) | S(8) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Colonnes | 0.148385 | 0.183446 | 0.125195 | 0.099800 | 0.809 | 1.185 | 1.487 |
| Lignes | 0.132493 | 0.118853 | 0.087667 | 0.061901 | 1.115 | 1.511 | 2.140 |

Comparaison qualitative attendue :
- Decoupage lignes : communication d'assemblage simple (gather du resultat local), souvent plus efficace.
- Decoupage colonnes : necessite une reduction globale sur un vecteur complet, parfois plus couteux en communication.
- Les mesures confirment cette tendance : le decoupage par lignes est nettement meilleur ici.

## 3. Questions de cours (Amdahl / Gustafson)

Donnee : fraction parallele `p = 0.9`.

### 3.1 Acceleration maximale (Amdahl, n -> inf)

`S_max = 1 / (1 - p) = 1 / 0.1 = 10`.

Alice ne pourra pas depasser un facteur 10 sur ce jeu de donnees, meme avec un tres grand nombre de noeuds.

### 3.2 Nombre de noeuds raisonnable

`S(n) = 1 / ((1 - p) + p/n) = 1 / (0.1 + 0.9/n)`.

Valeurs utiles :
- `n = 8`  -> `S = 4.71`
- `n = 16` -> `S = 6.40`
- `n = 32` -> `S = 7.80`
- `n = 64` -> `S = 8.77`

Le rendement marginal baisse vite apres 16-32 noeuds. Une plage raisonnable est donc souvent 16 a 32 noeuds pour limiter le gaspillage CPU.

### 3.3 Loi de Gustafson si Alice observe S=4 puis double les donnees

Si Alice mesure `S(n)=4` sur `n` noeuds, alors :
- `S = n - alpha (n - 1)` avec `alpha` fraction strictement sequentielle.
- Donc `alpha = (n - S)/(n - 1)`.

En doublant la taille du probleme et en supposant que la partie parallele scale lineairement, l'acceleration de Gustafson reste essentiellement liee a `n` et `alpha` (pas plafonnee comme Amdahl pour un probleme fixe). On conserve :
- `S_G(n) = n - alpha (n - 1)`

Exemple si `n=8` et `S=4` observe :
- `alpha = (8 - 4)/(8 - 1) = 4/7`
- `S_G(8) = 8 - (4/7)*7 = 4`

L'idee cle est que Gustafson favorise l'augmentation de taille de probleme : on peut maintenir une bonne efficacite quand la charge parallele augmente avec les ressources.

## 4. Notes d'implementation

- Le script `tp2/matvec.py` contenait un risque de blocage pour `--check` en parallele ; corrige avec un `Bcast` appele par tous les rangs.
- `tp2/mandelbrot.py` a ete rendu tolerant a l'absence de `matplotlib/Pillow` : le calcul MPI et les temps sont disponibles meme si la sauvegarde image est ignoree.
