CC=gcc

SRCDIR=../src
OBJDIR=./obj
BINDIR=../bin
OBJFILES = mlp.o
OBJ = $(patsubst %,$(OBJDIR)/%,$(OBJFILES))

#----------------
# RULES
#----------------

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(OBJDIR)
	$(CC) -c -o $@ $<

$(BINDIR)/mlp: $(OBJ)
	$(CC) -o $@ $^

.PHONY: clean
clean:
	@rm -f $(OBJDIR)/*.o
	@if [ -d $(OBJDIR) ]; then rmdir $(OBJDIR); fi
