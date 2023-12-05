#!/usr/bin/python

from monitor import analyzer

def main():
    xc = analyzer(history_size_threshold=100000, sleeptime=0.001, use_redis=True)
    xc.run_history()

if __name__ == '__main__':
    main()
