import trec_main

def benchmark():
	k1 = 0.0
	b = 0.0
	while(k1 <= 2.0):
		while(b <= 1.0):
			print("Running with b = {0}, k1 = {1}".format(k1, b))
			trec_main.run(k1=k1, b=b)
			b += 0.1
		k1 += 0.1
	print("Benchmark finished")

if __name__ == "__main__":
	benchmark()
