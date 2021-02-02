import warnings
from subprocess import Popen, PIPE
from codecs import getdecoder
from ..spot.promela import Parser


def run_spot(formula: str, debug=False):
    """
    A function to call the spot toolbox and run interpet the output which is in NeverClaim format
    For our prupose these are the relevant flags:
    1. --complete : Output a complete a automaton
    2. -- ba : Specifically mention the output to be a Büchi Automaton
    3. --state-based-acceptance : The default is transition based rep (usually a smaller rep) but we over ride that
    using this flag
    4. --deterministic : prefer a deterministic automaton
    5. --spin : Flag to make sure that the output is in NeverClaim format
    6. --formula : The input formula
    =============================================================================================
    Guide :

    Usage: ltl2tgba [OPTION...] [FORMULA...]
    Translate linear-time formulprint(f"==========================================")as (LTL/PSL) into various types of automata.

    By default it will apply all available optimizations to output the smallest
    Transition-based Generalized Büchi Automata, output in the HOA format.
    If multiple formulas are supplied, several automata will be output.

     Input options:
      -f, --formula=STRING       process the formula STRING
      -F, --file=FILENAME[/COL]  process each line of FILENAME as a formula; if COL
                                 is a positive integer, assume a CSV file and read
                                 column COL; use a negative COL to drop the first
                                 line of the CSV file
          --lbt-input            read all formulas using LBT's prefix syntax
          --lenient              parenthesized blocks that cannot be parsed as
                                 subformulas are considered as atomic properties
          --negate               negate each formula

     Output automaton type:
      -B, --ba                   Büchi Automaton (implies -S)
          --cobuchi, --coBuchi   automaton with co-Büchi acceptance (will
                                 recognize a superset of the input language if not
                                 co-Büchi realizable)
      -C, --complete             output a complete automaton
      -G, --generic              any acceptance condition is allowed
      -M, --monitor              Monitor (accepts all finite prefixes of the given
                                 property)
      -p, --colored-parity[=any|min|max|odd|even|min odd|min even|max odd|max
          even]                  colored automaton with parity acceptance
      -P, --parity[=any|min|max|odd|even|min odd|min even|max odd|max even]
                                 automaton with parity acceptance
      -S, --state-based-acceptance, --sbacc
                                 define the acceptance using states
          --tgba                 Transition-based Generalized Büchi Automaton
                                 (default)
      -U, --unambiguous          output unambiguous automata

     Output format:
      -8, --utf8                 enable UTF-8 characters in output (ignored with
                                 --lbtt or --spin)
          --check[=PROP]         test for the additional property PROP and output
                                 the result in the HOA format (implies -H).  PROP
                                 may be some prefix of 'all' (default),
                                 'unambiguous', 'stutter-invariant',
                                 'stutter-sensitive-example', 'semi-determinism',
                                 or 'strength'.
      -d, --dot[=1|a|A|b|B|c|C(COLOR)|e|E|f(FONT)|h|k|K|n|N|o|r|R|s|t|u|v|y|+INT|<INT|#]                                                          GraphViz's format.
                                 Add letters for (1) force numbered states, (a)
                                 show acceptance condition (default), (A) hide
                                 acceptance condition, (b) acceptance sets as
                                 bullets, (B) bullets except for Büchi/co-Büchi
                                 automata, (c) force circular nodes, (C) color
                                 nodes with COLOR, (d) show origins when known, (e)
                                 force elliptic nodes, (E) force rEctangular nodes,
                                 (f(FONT)) use FONT, (g) hide edge labels, (h)
                                 horizontal layout, (k) use state labels when
                                 possible, (K) use transition labels (default), (n)
                                 show name, (N) hide name, (o) ordered transitions,
                                 (r) rainbow colors for acceptance sets, (R) color
                                 acceptance sets by Inf/Fin, (s) with SCCs, (t)
                                 force transition-based acceptance, (u) hide true
                                 states, (v) vertical layout, (y) split universal
                                 edges by color, (+INT) add INT to all set numbers,
                                 (<INT) display at most INT states, (#) show
                                 internal edge numbers
      -H, --hoaf[=1.1|i|k|l|m|s|t|v]   Output the automaton in HOA format
                                 (default).  Add letters to select (1.1) version
                                 1.1 of the format, (i) use implicit labels for
                                 complete deterministic automata, (s) prefer
                                 state-based acceptance when possible [default],
                                 (t) force transition-based acceptance, (m) mix
                                 state and transition-based acceptance, (k) use
                                 state labels when possible, (l) single-line
                                 output, (v) verbose properties
          --lbtt[=t]             LBTT's format (add =t to force transition-based
                                 acceptance even on Büchi automata)
          --name=FORMAT          set the name of the output automaton
      -o, --output=FORMAT        send output to a file named FORMAT instead of
                                 standard output.  The first automaton sent to a
                                 file truncates it unless FORMAT starts with '>>'.
      -q, --quiet                suppress all normal output
      -s, --spin[=6|c]           Spin neverclaim (implies --ba).  Add letters to
                                 select (6) Spin's 6.2.4 style, (c) comments on
                                 states
          --stats=FORMAT, --format=FORMAT
                                 output statistics about the automaton

     Any FORMAT string may use the following interpreted sequences:
      %<                         the part of the line before the formula if it
                                 comes from a column extracted from a CSV file
      %>                         the part of the line after the formula if it comes
                                 from a column extracted from a CSV file
      %%                         a single %
      %a                         number of acceptance sets
      %c, %[LETTERS]c            number of SCCs; you may filter the SCCs to count
                                 using the following LETTERS, possibly
                                 concatenated: (a) accepting, (r) rejecting, (c)
                                 complete, (v) trivial, (t) terminal, (w) weak,
                                 (iw) inherently weak. Use uppercase letters to
                                 negate them.
      %d                         1 if the output is deterministic, 0 otherwise
      %e                         number of reachable edges
      %f                         the formula, in Spot's syntax
      %F                         name of the input file
      %g, %[LETTERS]g            acceptance condition (in HOA syntax); add brackets
                                 to print an acceptance name instead and LETTERS to
                                 tweak the format: (0) no parameters, (a)
                                 accentuated, (b) abbreviated, (d) style used in
                                 dot output, (g) no generalized parameter, (l)
                                 recognize Street-like and Rabin-like, (m) no main
                                 parameter, (p) no parity parameter, (o) name
                                 unknown acceptance as 'other', (s) shorthand for
                                 'lo0'.
      %h                         the automaton in HOA format on a single line (use
                                 %[opt]h to specify additional options as in
                                 --hoa=opt)
      %L                         location in the input file
      %m                         name of the automaton
      %n                         number of nondeterministic states in output
      %p                         1 if the output is complete, 0 otherwise
      %r                         wall-clock time elapsed in seconds (excluding
                                 parsing)
      %R, %[LETTERS]R            CPU time (excluding parsing), in seconds; Add
                                 LETTERS to restrict to(u) user time, (s) system
                                 time, (p) parent process, or (c) children
                                 processes.
      %s                         number of reachable states
      %t                         number of reachable transitions
      %u, %[e]u                  number of states (or [e]dges) with universal
                                 branching
      %u, %[LETTER]u             1 if the automaton contains some universal
                                 branching (or a number of [s]tates or [e]dges with
                                 universal branching)
      %w                         one word accepted by the output automaton
      %x, %[LETTERS]x            number of atomic propositions declared in the
                                 automaton;  add LETTERS to list atomic
                                 propositions with (n) no quoting, (s) occasional
                                 double-quotes with C-style escape, (d)
                                 double-quotes with C-style escape, (c)
                                 double-quotes with CSV-style escape, (p) between
                                 parentheses, any extra non-alphanumeric character
                                 will be used to separate propositions

     Simplification goal:
      -a, --any                  no preference, do not bother making it small or
                                 deterministic
      -D, --deterministic        prefer deterministic automata (combine with
                                 --generic to be sure to obtain a deterministic
                                 automaton)
          --small                prefer small automata (default)

     Simplification level:
          --high                 all available optimizations (slow, default)
          --low                  minimal optimizations (fast)
          --medium               moderate optimizations

     Miscellaneous options:
      -x, --extra-options=OPTS   fine-tuning options (see spot-x (7))
          --help                 print this help
          --version              print program version

    Mandatory or optional arguments to long options are also mandatory or optional
    for any corresponding short options.

    :return:
    """
    if not isinstance(formula, str):
        warnings.warn("Please make sure the input formula is of type string.")

    process = Popen(["ltl2tgba", "--complete", "--ba", "--state-based-acceptance",
                               "--deterministic", "--spin", f"--formula={formula}"], stdout=PIPE)
    (raw_output, err) = process.communicate()
    if err:
        print(err)
        warnings.warn("The process did not run properly. Please check if spot is installed correctly, and the formula "
                      "is a string")

    ascii_decoder = getdecoder("ascii")
    (processed_output, _) = ascii_decoder(raw_output)
    if debug:
        print(f"Output of Spot for the formula : {formula}")
        print(f"==========================================")
        print(processed_output)
        print(f"==========================================")

    return processed_output


def parse_ltl(formula: str, debug=False):
    spot_output = run_spot(formula, debug)
    parser = Parser(spot_output)
    edges = parser.parse()
    if debug:
        print(edges)


if __name__ == "__main__":
    parse_ltl("!b U c")