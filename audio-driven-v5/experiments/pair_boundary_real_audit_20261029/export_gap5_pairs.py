"""Export the five-second result under new IDs; retain the three-second media."""
import export_fixed_pairs as export
if __name__=='__main__':
    export.RESULTS=export.EXP/'gap5_pairs'
    export.OUT=export.EXP/'review/gap5_pairs'
    export.MEDIA_DIR='gap5_pairs'
    export.ID_PREFIX='G'
    export.main()
