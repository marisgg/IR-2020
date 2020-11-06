#!/usr/bin/env python3

import argparse

def main():
	parser = argparse.ArgumentParser(description="TREC-COVID document ranker CLI")
	parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
	args = parser.parse_args()

if __name__ == '__main__':
	main()
