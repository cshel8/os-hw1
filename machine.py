"""machine.py: a register-level simulator of the small 16-bit accumulator machine.

Run it from a terminal in the folder that holds this file:
    python machine.py            (Windows)   or   python3 machine.py   (macOS / Linux)
    python machine.py --check    runs the built-in self-test
    python machine.py --broken   runs the interrupt demo with a handler that forgets to save the AC

Your work is marked with TODO. Everything else already works: run the file before you change
anything and it will stop at the first TODO it reaches and tell you which one it is.
"""

import sys  # sys gives us the command-line words (sys.argv) and a clean way to stop the program (sys.exit)

# ---------------------------------------------------------------------------
# 1. The machine's fixed facts
# ---------------------------------------------------------------------------
MEMORY_WORDS = 4096  # a 12-bit address field can name 2**12 = 4,096 different words, so memory has exactly that many
WORD_MASK = 0xFFFF  # every word is 16 bits wide; AND-ing with 0xFFFF throws away anything above bit 15
SIGN_BIT = 0x8000  # in sign-magnitude, the leftmost bit (bit 15) is the sign: 0 means positive, 1 means negative
MAGNITUDE_MASK = 0x7FFF  # the other 15 bits hold the size of the number, so the largest magnitude is 32,767

OP_HALT = 0x0  # opcode 0000: stop the machine (a stop code added for this simulator)
OP_LOAD = 0x1  # opcode 0001: load the AC from memory
OP_STORE = 0x2  # opcode 0010: store the AC to memory
OP_ADD = 0x5  # opcode 0101: add a word from memory to the AC
OP_IRET = 0xF  # opcode 1111: return from interrupt (PC comes back from the save word; interrupts turn back on)

SAVE_WORD = 0x000  # when an interrupt is accepted, the hardware saves the PC here (a one-word stand-in for the control stack)
HANDLER_START = 0x100  # the hardware then jumps here, where the interrupt handler's first instruction lives
MAX_STEPS = 200  # a safety net: if a buggy program never halts, the simulator stops after this many instructions


class NotWrittenYet(Exception):  # a custom error the starter raises wherever a TODO has not been written yet
    pass  # it needs no extra behavior; its name and message say everything


# ---------------------------------------------------------------------------
# 2. Sign-magnitude helpers: turn a 16-bit word into a Python int and back
# ---------------------------------------------------------------------------
def to_int(word):  # word: a 16-bit pattern read from memory, such as 0x8007
    magnitude = word & MAGNITUDE_MASK  # keep only the low 15 bits: how big the number is
    if word & SIGN_BIT:  # if the sign bit is 1 the number is negative
        return -magnitude  # so 0x8007 becomes -7
    return magnitude  # otherwise it is positive, so 0x0019 becomes 25


def to_word(value):  # value: an ordinary Python int such as -7 or 32
    if abs(value) > MAGNITUDE_MASK:  # a result that needs more than 15 bits cannot be stored
        raise OverflowError(f"{value} does not fit in 15 bits plus a sign bit")  # stop loudly instead of storing a wrong number
    if value < 0:  # negative numbers get the sign bit switched on
        return SIGN_BIT | (-value)  # -7 becomes 0x8000 | 0x0007 = 0x8007
    return value  # zero and positive numbers are stored as they are


def hex4(value):  # formats a word as exactly four hex digits, the way the chapter writes words
    return f"{value & WORD_MASK:04X}"  # 32 becomes "0020"


def hex3(value):  # formats an address as exactly three hex digits
    return f"{value & 0xFFF:03X}"  # 0x250 becomes "250"


# ---------------------------------------------------------------------------
# 3. The machine itself
# ---------------------------------------------------------------------------
class Machine:  # one object of this class is one complete computer: memory plus processor registers

    def __init__(self, program, start):  # program: {address: word}; start: the address of the first instruction
        self.memory = [0] * MEMORY_WORDS  # every word starts as 0000 before the program is loaded
        for address, word in program.items():  # copy each (address, word) pair of the program into memory
            self.memory[address] = word & WORD_MASK  # & WORD_MASK guarantees nothing wider than 16 bits gets in
        self.pc = start  # program counter: the address of the NEXT instruction to fetch
        self.ir = 0  # instruction register: holds the instruction currently being executed
        self.ac = 0  # accumulator: the one data register this machine has
        self.mar = 0  # memory address register: the address sent to memory for the next read or write
        self.mbr = 0  # memory buffer register: the word just read from memory, or about to be written to it
        self.halted = False  # becomes True when a HALT instruction executes
        self.interrupts_enabled = True  # the hardware accepts interrupts only while this is True
        self.pending_interrupt = False  # set to True when a device (here, a timer) raises an interrupt
        self.interrupt_after = set()  # instruction counts after which the timer raises an interrupt, e.g. {2}
        self.executed = 0  # how many instructions have finished their execute stage so far
        self.reads = 0  # how many memory reads have happened (fetches count too)
        self.writes = 0  # how many memory writes have happened
        self.rows = []  # the trace: one row per stage, saved so it can be printed or checked later

    def read(self, address):  # every memory read goes through here so it is always done the same way
        self.mar = address  # the processor puts the address in the MAR
        self.mbr = self.memory[address]  # memory answers by placing that word in the MBR
        self.reads += 1  # count it
        return self.mbr  # hand the word back to whoever asked

    def write(self, address, word):  # every memory write goes through here
        self.mar = address  # the address goes in the MAR
        self.mbr = word & WORD_MASK  # the data goes in the MBR
        self.memory[address] = self.mbr  # memory copies the MBR into that word
        self.writes += 1  # count it

    def record(self, stage):  # saves one trace row showing every register after a stage
        self.rows.append((stage, hex3(self.pc), hex4(self.ir), hex4(self.ac), hex3(self.mar), hex4(self.mbr)))  # a tuple of formatted strings

    def fetch(self):  # the fetch stage: bring the next instruction into the IR
        self.ir = self.read(self.pc)  # MAR <- PC, MBR <- memory[PC], then IR <- MBR
        self.pc = (self.pc + 1) & 0xFFF  # the PC moves on at once, so during execute it already names the next instruction
        self.record("fetch")  # save the registers as they stand after the fetch

    def execute(self):  # the execute stage: decode the IR and do what it says
        opcode = self.ir >> 12  # the top 4 bits of the 16-bit instruction are the opcode
        address = self.ir & 0xFFF  # the low 12 bits are the address the instruction refers to
        if opcode == OP_LOAD:  # 0001: load AC from memory
            self.ac = self.read(address)  # MAR <- address, MBR <- memory[address], AC <- MBR
        elif opcode == OP_STORE:  # 0010: store AC to memory
            # TODO 1: the AC's value must end up in memory at `address`. Use self.write (it sets the MAR and MBR for you).
            self.write(address, self.ac)
        elif opcode == OP_ADD:  # 0101: add memory to AC
            # TODO 2: read the operand with self.read, add it to the AC as SIGNED numbers (to_int turns a word into an int),
            #         then turn the sum back into a 16-bit word (to_word) and put it in self.ac.
            self.ac = to_word(to_int(self.ac) + to_int(self.read(address)))
        elif opcode == OP_HALT:  # 0000: stop
            # TODO 3: tell the run loop to stop (look at what run() checks).
            self.halted = True  # tell the run loop to stop
        elif opcode == OP_IRET:  # 1111: return from interrupt
            # TODO 5: put back the PC that the interrupt stage saved in memory word SAVE_WORD (use self.read), then turn interrupts back on.
            self.pc = self.read(SAVE_WORD)
            self.interrupts_enabled = True
        else:  # any other opcode is not part of this machine
            raise ValueError(f"unknown opcode {opcode:X} in instruction {hex4(self.ir)}")  # stop with a clear message
        self.executed += 1  # one more instruction has completed
        self.record("execute")  # save the registers as they stand after the execute
        if self.executed in self.interrupt_after:  # the timer is set to go off after this instruction
            self.pending_interrupt = True  # raise the interrupt; it is noticed in the interrupt stage below

    def interrupt_stage(self):  # runs after every execute: is there an interrupt to accept?
        if not (self.pending_interrupt and self.interrupts_enabled and not self.halted):  # nothing waiting, or interrupts are off
            return  # so the processor simply goes on to the next fetch
        # TODO 4: accept the interrupt. In order:
        #   (a) it is no longer pending;  (b) turn interrupts off so the handler cannot itself be interrupted;
        #   (c) save the PC into memory word SAVE_WORD (use self.write);  (d) load the PC with HANDLER_START;
        #   (e) call self.record("interrupt") so the trace shows this stage.
        self.pending_interrupt = False
        self.interrupts_enabled = False
        self.write(SAVE_WORD, self.pc)
        self.pc = HANDLER_START
        self.record("interrupt")

    def run(self):  # the instruction cycle: fetch, execute, check for interrupts, repeat
        while not self.halted:  # keep cycling until a HALT executes
            if self.executed >= MAX_STEPS:  # safety net for programs that never halt
                raise RuntimeError("the program did not halt; is a HALT (0000) missing?")  # stop instead of looping forever
            self.fetch()  # stage 1
            self.execute()  # stage 2
            self.interrupt_stage()  # stage 3 (does nothing when no interrupt is waiting)
        return self  # returning the machine lets a caller write Machine(...).run().memory

    def print_trace(self):  # prints the trace as a table the answer sheet can hold
        print(f"{'step':>4}  {'stage':<9} {'PC':>3}  {'IR':>4}  {'AC':>4}  {'MAR':>3}  {'MBR':>4}")  # the header row
        for number, (stage, pc, ir, ac, mar, mbr) in enumerate(self.rows, start=1):  # number every row from 1
            print(f"{number:>4}  {stage:<9} {pc:>3}  {ir:>4}  {ac:>4}  {mar:>3}  {mbr:>4}")  # one aligned row per stage
        print(f"memory reads: {self.reads}   memory writes: {self.writes}")  # the totals Problem 1 asks about


# ---------------------------------------------------------------------------
# 4. Programs
# ---------------------------------------------------------------------------
PROBLEM_1 = {  # the program from Problem 1 of the written part
    0x250: 0x13A0,  # load AC from 3A0
    0x251: 0x53A1,  # add word 3A1 to AC
    0x252: 0x53A2,  # add word 3A2 to AC
    0x253: 0x23A3,  # store AC to 3A3
    0x254: 0x0000,  # halt
    0x3A0: 0x0019,  # data: +25
    0x3A1: 0x000E,  # data: +14
    0x3A2: 0x8007,  # data: -7 in sign-magnitude
    0x3A3: 0x0000,  # the result goes here
}  # end of PROBLEM_1

HANDLER_SAFE = {  # an interrupt handler that counts timer ticks and protects the AC
    0x100: 0x2001,  # store AC to 001: save the interrupted program's AC before using the AC
    0x101: 0x10F0,  # load AC from 0F0: the tick counter
    0x102: 0x50F1,  # add 0F1: the constant 1
    0x103: 0x20F0,  # store AC to 0F0: counter + 1
    0x104: 0x1001,  # load AC from 001: put the interrupted program's AC back
    0x105: 0xF000,  # IRET: return to the interrupted program
    0x0F0: 0x0000,  # the tick counter starts at 0
    0x0F1: 0x0001,  # the constant 1
}  # end of HANDLER_SAFE

HANDLER_BROKEN = {  # the same handler with the save and restore removed
    0x100: 0x10F0,  # load AC from 0F0: this overwrites the interrupted program's AC!
    0x101: 0x50F1,  # add the constant 1
    0x102: 0x20F0,  # store the new count
    0x103: 0xF000,  # IRET
    0x0F0: 0x0000,  # the tick counter starts at 0
    0x0F1: 0x0001,  # the constant 1
}  # end of HANDLER_BROKEN

CHECK_PROGRAM = {  # a small program with known answers, used only by --check
    0x010: 0x1020,  # load AC from 020 (+9)
    0x011: 0x5021,  # add 021 (-4): AC = 5
    0x012: 0x2022,  # store AC to 022
    0x013: 0x5023,  # add 023 (-10): AC = -5
    0x014: 0x2024,  # store AC to 024
    0x015: 0x0000,  # halt
    0x020: 0x0009,  # +9
    0x021: 0x8004,  # -4
    0x022: 0x0000,  # result 1 goes here
    0x023: 0x800A,  # -10
    0x024: 0x0000,  # result 2 goes here
}  # end of CHECK_PROGRAM


def self_check():  # runs the check program with and without an interrupt and compares against known answers
    plain = Machine(CHECK_PROGRAM, start=0x010).run()  # no interrupts
    assert plain.memory[0x022] == 0x0005, "word 022 should be 0005 (9 + -4)"  # checks LOAD, ADD and STORE
    assert plain.memory[0x024] == 0x8005, "word 024 should be 8005 (5 + -10 = -5)"  # checks a negative sign-magnitude result
    assert plain.reads == 9 and plain.writes == 2, "expected 9 reads and 2 writes"  # 6 fetches + 3 operand reads, 2 stores
    ticked = Machine({**CHECK_PROGRAM, **HANDLER_SAFE}, start=0x010)  # the same program with the safe handler loaded
    ticked.interrupt_after = {3}  # the timer goes off after the third instruction (the first STORE)
    ticked.run()  # run it
    assert ticked.memory[0x0F0] == 0x0001, "the handler should have counted exactly one tick"  # checks the interrupt stage and IRET
    assert ticked.memory[0x024] == 0x8005, "the interrupt must not change the program's result"  # checks the AC was protected
    print("self-check passed: LOAD, ADD, STORE, HALT, the interrupt stage and IRET all behave correctly")  # good news


def main():  # decides what to run from the command-line words
    try:  # any TODO that is still missing raises NotWrittenYet; the except below turns that into a friendly message
        run_everything()  # the real work
    except NotWrittenYet as missing:  # a TODO was reached
        print(f"\nStopped: {missing}. Write that part, then run the program again.")  # say exactly which TODO


def run_everything():  # runs the self-check or the two demo runs, depending on the command-line words
    if "--check" in sys.argv:  # python machine.py --check
        self_check()  # run the self-test
        return  # and stop
    print("Problem 1 program, no interrupts")  # a title for the first run
    first = Machine(PROBLEM_1, start=0x250)  # build the machine and load Problem 1
    try:  # run it; if a TODO is reached, still show the trace so far before stopping
        first.run()  # fetch, execute, interrupt check, until HALT
    finally:  # runs whether or not run() raised an error
        first.print_trace()  # show every stage that did happen
    print(f"word 3A3 = {hex4(first.memory[0x3A3])} ({to_int(first.memory[0x3A3])})")  # the result in hex and decimal
    handler = HANDLER_BROKEN if "--broken" in sys.argv else HANDLER_SAFE  # pick which handler to load
    name = "BROKEN handler (no save/restore)" if handler is HANDLER_BROKEN else "safe handler"  # a label for the output
    print(f"\nProblem 1 program, timer interrupt after instruction 2, {name}")  # a title for the second run
    second = Machine({**PROBLEM_1, **handler}, start=0x250)  # the same program with a handler loaded beside it
    second.interrupt_after = {2}  # the timer goes off after the second instruction completes
    try:  # run it; if a TODO is reached, still show the trace so far
        second.run()  # fetch, execute, interrupt check, until HALT
    finally:  # runs whether or not run() raised an error
        second.print_trace()  # show every stage that did happen
    print(f"word 3A3 = {hex4(second.memory[0x3A3])} ({to_int(second.memory[0x3A3])})   ticks counted = {to_int(second.memory[0x0F0])}")  # results


if __name__ == "__main__":  # true only when this file is run directly, not when another file imports it
    main()  # start the program
