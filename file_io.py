import codecs
import os

def create_file(file_address, mode = 'w'):
	fout = codecs.open(file_address, mode, 'utf-8')
	return fout

def read_file(file_address, mode = 'r'):
	fout = codecs.open(file_address, mode, 'utf-8')
	return fout

def write_line(file_pointer, line):
	file_pointer.write(str(line))
