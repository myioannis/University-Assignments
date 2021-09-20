all: search cluster

search: InputSEARCH.o Image.o LSH.o Technique.o Exact.o Utils.o MainSEARCH.o Hypercube.o Cluster.o Input.h Image.h Technique.h LSH.h Exact.h Utils.h Hypercube.h Cluster.h
	g++ -o search InputSEARCH.o Image.o Technique.o Utils.o MainSEARCH.o LSH.o Exact.o Hypercube.o Cluster.o -O3

cube: InputHYPERCUBE.o Image.o LSH.o Technique.o Exact.o Utils.o MainHYPERCUBE.o Hypercube.o Cluster.o Input.h Image.h Technique.h LSH.h Exact.h Utils.h Hypercube.h Cluster.h
	g++ -o cube InputHYPERCUBE.o Image.o Technique.o Utils.o MainHYPERCUBE.o LSH.o Exact.o Hypercube.o Cluster.o -O3

cluster: InputCLUSTER.o Image.o LSH.o Technique.o Exact.o Utils.o MainCLUSTER.o Hypercube.o Cluster.o Input.h Image.h Technique.h LSH.h Exact.h Utils.h Hypercube.h Cluster.h
	g++ -o cluster InputCLUSTER.o Image.o Technique.o Utils.o MainCLUSTER.o LSH.o Exact.o Hypercube.o Cluster.o -O3

InputSEARCH.o: Input.cpp Input.h Image.h LSH.h Utils.h Exact.h Cluster.h
	g++ -c Input.cpp -o InputSEARCH.o -D SEARCH_FLAG 

InputHYPERCUBE.o: Input.cpp Input.h Image.h LSH.h Utils.h Exact.h Cluster.h
	g++ -c Input.cpp -o InputHYPERCUBE.o -D HYPERCUBE_FLAG

InputCLUSTER.o: Input.cpp Input.h Image.h LSH.h Utils.h Exact.h Cluster.h
	g++ -c Input.cpp -o InputCLUSTER.o -D CLUSTER_FLAG

Image.o: Image.cpp Input.h Image.h Utils.h
	g++ -c Image.cpp -O3

Utils.o: Utils.cpp Input.h Image.h Utils.h
	g++ -c Utils.cpp -O3

MainSEARCH.o: Main.cpp Input.h Image.h Utils.h Cluster.h
	g++ -c Main.cpp -o MainSEARCH.o -D SEARCH_FLAG -O3

MainHYPERCUBE.o: Main.cpp Input.h Image.h Utils.h Cluster.h
	g++ -c Main.cpp -o MainHYPERCUBE.o -D HYPERCUBE_FLAG -O3

MainCLUSTER.o: Main.cpp Input.h Image.h Utils.h Cluster.h
	g++ -c Main.cpp -o MainCLUSTER.o -D CLUSTER_FLAG -O3

LSH.o: LSH.cpp Image.h LSH.h Technique.h Utils.h
	g++ -c LSH.cpp -O3

Technique.o: Technique.cpp Image.h Technique.h
	g++ -c Technique.cpp -O3

Exact.o: Exact.cpp Exact.h Technique.h Input.h
	g++ -c Exact.cpp -O3

Hypercube.o: Hypercube.cpp Input.h Image.h LSH.h Technique.h Exact.h Utils.h Hypercube.h
	g++ -c Hypercube.cpp -O3

Cluster.o: Cluster.cpp Input.h Image.h LSH.h Technique.h Exact.h Utils.h Hypercube.h Cluster.h
	g++ -c Cluster.cpp -O3

clean:
	rm -r *.o search cluster