
# Compiler & flags
CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors

# Math library
LDFLAGS = -lm

# link the executable 
symnmf: symnmf.o
	$(CC) -o symnmf symnmf.o $(LDFLAGS)

# compile symnmf.c to object file
symnmf.o: symnmf.c symnmf.h
	$(CC) -c symnmf.c $(CFLAGS)

# remove the build artefacts
clean:
	rm -f symnmf symnmf.o
