/* kokoSD 2.4: what Koko remembers about you (engine/memory.c). */
#ifndef KOKO_MEMORY_H
#define KOKO_MEMORY_H
int mem_learn(const char *s, int n);              /* note what a message says about you; 1 when something new */
int mem_text(int you, char *out, int max);        /* you 0: "My name is Sam. ...", 1: "Your name is Sam. ...", 2: a line each */
int mem_count(void);
void mem_forget(void);
int mem_load(void);                               /* /koko/memory.txt: the number of facts read */
int mem_save(void);                               /* 1 when written */
#define MEM_BUF 720
extern char mem_buf[MEM_BUF];                    /* a text buffer: the file, the memory page, a hint question */
extern int mem_on;                                /* the menu's Memory: On / Off */
#endif
