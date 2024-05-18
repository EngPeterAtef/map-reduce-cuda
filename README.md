# MapReduce with CUDA


## How to run

### 1. CUDA


```bash
cd ./CUDA/
```
1. include your desired configuration in the map_reduce.cu  
for wordcount, include the following line:
```c
#include "wordcount.cuh"
```
for kmeans, include the following line:
```c
#include "kmeans.cuh"
```

2. compile the code with the following command:  
Using the makefile:  

wordcount
```bash
make wordcount
```
kmeans
```bash
make kmeans
```
or manually:  
wordcount
```bash
nvcc -O3 -dc map_reduce.cu
nvcc -O3 -o wordcount map_reduce.o
```
kmeans  
```bash
nvcc -O3 -dc map_reduce.cu
nvcc -O3 -o kmeans map_reduce.o
```

3. run the code with the following command:  

wordcount
```bash
./wordcount ../data/WordCount word_count_input1.txt
```
kmeans
```bash
./kmeans ../data/Kmeans/s1_2_15.txt
```

4. Output will be save in the file {input_name}_output.txt

### 2. CPU

```bash
cd ./CPU/
```
1. include your desired configuration in the map_reduce.cpp  
for wordcount, include the following line:
```c
#include "wordcount.hpp"
```
for kmeans, include the following line:
```c
#include "kmeans.hpp"
```

2. compile the code with the following command:  
Using the makefile:  

wordcount
```bash
make wordcount
```
kmeans
```bash
make kmeans
```
or manually:  
wordcount
```bash
g++ -o wordcount map_reduce.cpp
```
kmeans  
```bash
g++ -o kmeans map_reduce.cpp
```

3. run the code with the following command:  

wordcount
```bash
./wordcount ../data/WordCount/word_count_input1.txt
```
kmeans
```bash
./kmeans ../data/Kmeans/s1_2_15.txt
```

4. Output will be save in the file {input_name}_output.txt