SHELL := /bin/bash

TARGET:=papiutil.o
SOURCES:=$(wildcard ./src/*.c)
HEADERS:=$(wildcard include/*.h)
OBJECTS:=$(SOURCES:.c=.o)

INC:=-Iinclude $(PAPI_INC)

CFLAGS ?= -Wall -Wextra -O2 -fopenmp

all: $(OBJECTS)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) $(INC) -c $< -o $@

.PHONY: clean
clean:
	$(RM) $(OBJECTS)
