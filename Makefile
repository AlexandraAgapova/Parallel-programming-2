all:
	g++ -O2 -std=c++17 -pthread task1.cpp -o task1
	g++ -O2 -fopenmp task2.cpp -o task2

clean:
	rm -f task1 task2
