"""Run the classification pipeline.

Thin wrapper around linkages.pipeline.main() for running as a script.
Equivalent to: linkages-classify [args]
"""

from linkages.pipeline import main

if __name__ == "__main__":
    main()
