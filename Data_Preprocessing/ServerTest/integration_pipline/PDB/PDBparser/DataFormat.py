##!/usr/bin/python3



class DataFormat:
    """
    Create the readable format for the files offered by PDBparser, etc. *.ent, *.pdb, *.xml ...
    """
    @property
    def txt_based_data(self):
        """
        The file format according to the Atomic Coordinate Entry Format Description V3.3
        See: http://www.wwpdb.org/documentation/file-format-content/format33/v3.3.html
        :return: a dict with the record format in a text-based PDF file
        """
        pdb_format = {}
        pdb_format.update(HEADER=dict(classification=dict(start=[11], end=[50], datatype=str, description='Classifies the molecule(s).'),
                                    depDate=dict(start=[51], end=[59], datatype=str, description='Deposition date. This is the date the'),
                                    idCode=dict(start=[63], end=[66], datatype=str, description='This identifier is unique within the PDBparser.')))

        pdb_format.update(TITLE=dict(continuation=dict(start=[9], end=[10], datatype=str, description='Allows concatenation of records if necessary.'),
                                    title=dict(start=[11], end=[80], datatype=str, description='Title of the  experiment.')))

        pdb_format.update(COMPND=dict(continuation=dict(start=[8], end=[10], datatype=str, description='Allows concatenation of records if necessary.'),
                                    compound=dict(start=[11], end=[80], datatype=str, description='Description of the molecular components.')))

        pdb_format.update(SOURCE=dict(continuation=dict(start=[8], end=[10], datatype=str, description='Allows concatenation of records if necessary.'),
                                    srcName=dict(start=[11], end=[80], datatype=str, description='Description of the molecular components.')))

        pdb_format.update(KEYWDS=dict(continuation=dict(start=[9], end=[10], datatype=str, description='Allows concatenation of records if necessary.'),
                                    kwtwds=dict(start=[11], end=[79], datatype=str, description='Comma-separated list of keywords relevantto the entry.')))

        pdb_format.update(EXPDTA=dict(continuation=dict(start=[9], end=[10], datatype=str, description='Allows concatenation of records if necessary.'),
                                    technique=dict(start=[11], end=[79], datatype=str, description='The experimental technique(s) with optional comment describing the sample or experiment.')))

        pdb_format.update(AUTHOR=dict(continuation=dict(start=[9], end=[10], datatype=str, description='Allows concatenation of records if necessary.'),
                                    authorList=dict(start=[11], end=[79], datatype=str, description='List of the author names, separated by commas.')))

        pdb_format.update(REVDAT=dict(modNum=dict(start=[8], end=[10], datatype=int, description='Modification number.'),
                                    continuation=dict(start=[11], end=[12], datatype=str, description='Allows concatenation of multiple records.'),
                                    modDate=dict(start=[14], end=[22], datatype=str, description='Date of modification (or release  for  new entries)  in DD-MMM-YY format. This is not repeated on continued lines.'),
                                    modId=dict(start=[24], end=[27], datatype=str, description='ID code of this entry. This is not repeated on continuation lines.'),
                                    modType=dict(start=[32], end=[32], datatype=int, description='An integer identifying the type of    modification. For all  revisions, the modification type is listed as 1'),
                                    record=dict(start=[40, 47, 54, 61], end=[45, 52, 59, 66], datatype=str, description='Modification detail.'),
                                    # record=dict(start=47, end=52, datatype=str, description='Modification detail.'),
                                    # record=dict(start=54, end=58, datatype=str, description='Modification detail.'),
                                    # record=dict(start=61, end=66, datatype=str, description='Modification detail.')
                                      ))

        pdb_format.update(JRNL=dict(text=dict(start=[13], end=[79], datatype=str, description='See Details below.')))

        pdb_format.update(REMARK=dict(remarkNum=dict(start=[8], end=[10], datatype=int, description='Remark  number. It is not an error for remark n to exist in an entry when remark n-1 does not.'),
                                    empty=dict(start=[12], end=[79], datatype=str, description='Left  as white space in first line of each  new remark.')))

        pdb_format.update(OBSLTE=dict(continuation=dict(start=[9], end=[10], datatype=str, description='Allows concatenation of multiple recordsv'),
                                    repDate=dict(start=[12], end=[20], datatype=str, description='Date that this entry was replaced.'),
                                    idCode=dict(start=[22], end=[25], datatype=str, description='ID code of this entry.'),
                                    rIdCode=dict(start=[32, 37, 42, 47, 52, 57, 62, 67, 72], end=[35, 40, 45, 50, 55, 60, 65, 70, 75], datatype=str, description='ID code of entry that replaced this one.'),
                                    # rIdCode=dict(start=37, end=40, datatype=str, description='ID code of entry that replaced this one.'),
                                    # rIdCode=dict(start=42, end=45, datatype=str, description='ID code of entry that replaced this one.'),
                                    # rIdCode=dict(start=47, end=50, datatype=str, description='ID code of entry that replaced this one.'),
                                    # rIdCode=dict(start=52, end=55, datatype=str, description='ID code of entry that replaced this one.'),
                                    # rIdCode=dict(start=57, end=60, datatype=str, description='ID code of entry that replaced this one.'),
                                    # rIdCode=dict(start=62, end=65, datatype=str, description='ID code of entry that replaced this one.'),
                                    # rIdCode=dict(start=67, end=70, datatype=str, description='ID code of entry that replaced this one.'),
                                    # rIdCode=dict(start=72, end=75, datatype=str, description='ID code of entry that replaced this one.')
                                      ))

        pdb_format.update(SPLIT=dict(continuation=dict(start=[9], end=[10], datatype=str, description='Allows concatenation of multiple recordsv'),
                                    idCode=dict(start=[12], end=[15], datatype=str, description='ID code of related entry.'),
                                    # idCode=dict(start=17, end=20, datatype=str, description='ID code of related entry.'),
                                    # idCode=dict(start=22, end=25, datatype=str, description='ID code of related entry.'),
                                    # idCode=dict(start=27, end=30, datatype=str, description='ID code of related entry.'),
                                    # IdCode=dict(start=32, end=35, datatype=str, description='ID code of related entry.'),
                                    # IdCode=dict(start=37, end=40, datatype=str, description='ID code of related entry.'),
                                    # IdCode=dict(start=42, end=45, datatype=str, description='ID code of related entry.'),
                                    # IdCode=dict(start=47, end=50, datatype=str, description='ID code of related entry.'),
                                    # IdCode=dict(start=52, end=55, datatype=str, description='ID code of related entry.'),
                                    # IdCode=dict(start=57, end=60, datatype=str, description='ID code of related entry.'),
                                    # IdCode=dict(start=62, end=65, datatype=str, description='ID code of related entry.'),
                                    # IdCode=dict(start=67, end=70, datatype=str, description='ID code of related entry.'),
                                    # IdCode=dict(start=72, end=75, datatype=str, description='ID code of related entry.')
                                      ))

        pdb_format.update(SPRSDE=dict(continuation=dict(start=[9], end=[10], datatype=str, description='Allows for multiple ID codes.'),
                                    sprsdeDate=dict(start=[12], end=[20], datatype=str, description='Date this entry superseded the listed entries. This field is not copied on  continuations.'),
                                    idCode=dict(start=[22], end=[25], datatype=str, description='ID code of this entry. This field is  not copied on continuations.'),
                                    sIdCode=dict(start=[32, 37, 42, 47, 52, 57, 62, 67, 72], end=[35, 40, 45, 50, 55, 60, 65, 70, 75], datatype=str, description='ID code of a superseded entry.'),
                                    # sIdCode=dict(start=32, end=35, datatype=str, description='ID code of a superseded entry.'),
                                    # sIdCode=dict(start=37, end=40, datatype=str, description='ID code of a superseded entry.'),
                                    # sIdCode=dict(start=42, end=45, datatype=str, description='ID code of a superseded entry.'),
                                    # sIdCode=dict(start=47, end=50, datatype=str, description='ID code of a superseded entry.'),
                                    # sIdCode=dict(start=52, end=55, datatype=str, description='ID code of a superseded entry.'),
                                    # sIdCode=dict(start=57, end=60, datatype=str, description='ID code of a superseded entry.'),
                                    # sIdCode=dict(start=62, end=65, datatype=str, description='ID code of a superseded entry.'),
                                    # sIdCode=dict(start=67, end=70, datatype=str, description='ID code of a superseded entry.'),
                                    # sIdCode=dict(start=72, end=75, datatype=str, description='ID code of a superseded entry.')
                                      ))

        pdb_format.update(NUMMDL=dict(modelNumber=dict(start=[11], end=[14], datatype=int, description='Number of models.')
                                      ))

        pdb_format.update(CAVEAT=dict(continuation=dict(start=[9], end=[10], datatype=str, description='Allows for multiple ID codes.'),
                                    idCode=dict(start=[12], end=[15], datatype=str, description='PDBparser ID code of this entry.'),
                                    comment=dict(start=[20], end=[79], datatype=str, description='Free text giving the reason for the  CAVEAT.')
                                      ))

        pdb_format.update(MDLTYP=dict(continuation=dict(start=[9], end=[10], datatype=str, description='Allows for multiple ID codes.'),
                                    idCode=dict(start=[11], end=[80], datatype=str, description='Free Text providing  additional structural annotation.')
                                      ))

        pdb_format.update(DBREF=dict(idCode=dict(start=[8], end=[11], datatype=str, description='ID code of this entry.'),
                                    chainID=dict(start=[13], end=[13], datatype=str, description='Chain  identifier.'),
                                    seqBegin=dict(start=[15], end=[18], datatype=int, description='Initial sequence number of the PDBparser sequence segment.'),
                                    insertBegin=dict(start=[19], end=[19], datatype=str, description='Initial  insertion code of the PDBparser  sequence segment.'),
                                    seqEnd=dict(start=[21], end=[24], datatype=int, description='Ending sequence number of the PDBparser  sequence segment.'),
                                    insertEnd=dict(start=[25], end=[25], datatype=str, description='Ending insertion code of the PDBparser  sequence segment.'),
                                    database=dict(start=[27], end=[32], datatype=str, description='Sequence database name.'),
                                    dbAccession=dict(start=[34], end=[41], datatype=str, description='Sequence database accession code.'),
                                    dbIdCode=dict(start=[43], end=[54], datatype=str, description='Sequence  database identification code.'),
                                    dbseqBegin=dict(start=[56], end=[60], datatype=int, description='Initial sequence number of the database seqment.'),
                                    idbnsBeg=dict(start=[61], end=[61], datatype=str, description='Insertion code of initial residue of the segment, if PDBparser is the reference.'),
                                    dbseqEnd=dict(start=[63], end=[67], datatype=int, description='Ending sequence number of the database segment.'),
                                    dbinsEnd=dict(start=[68], end=[68], datatype=str, description='Insertion code of the ending residue of the segment, if PDBparser is the reference.'),
                                      ))

        pdb_format.update(SEQADV=dict(idCode=dict(start=[8], end=[11], datatype=str, description='ID code of this entry.'),
                                    resName=dict(start=[13], end=[15], datatype=str, description='Name of the PDBparser residue in conflict.'),
                                    chainID=dict(start=[17], end=[17], datatype=str, description='PDBparser  chain identifier.'),
                                    seqNum=dict(start=[19], end=[22], datatype=int, description='PDBparser  sequence number.'),
                                    iCode =dict(start=[23], end=[23], datatype=str, description='PDBparser insertion code.'),
                                    database=dict(start=[25], end=[28], datatype=str, description=''),
                                    dbAccession=dict(start=[30], end=[38], datatype=str, description='Sequence  database accession number.'),
                                    dbRes=dict(start=[40], end=[42], datatype=str, description='Sequence database residue name.'),
                                    dbSeq=dict(start=[44], end=[48], datatype=int, description='Sequence database sequence number.'),
                                    conflict=dict(start=[50], end=[70], datatype=str, description='Conflict comment.')
                                      ))

        pdb_format.update(MODRES=dict(idCode=dict(start=[8], end=[11], datatype=str, description='ID code of this entry.'),
                                    resName=dict(start=[13], end=[15], datatype=str, description='Residue name used in this entry.'),
                                    chainID=dict(start=[17], end=[17], datatype=str, description='Chain identifier.'),
                                    seqNum=dict(start=[19], end=[22], datatype=int, description='Sequence number.'),
                                    iCode =dict(start=[23], end=[23], datatype=str, description='Insertion code.'),
                                    stdRes=dict(start=[25], end=[27], datatype=str, description='Standard residue name.'),
                                    comment=dict(start=[30], end=[70], datatype=str, description='Description of the residue modification.')
                                      ))

        pdb_format.update(DBREF1=dict(idCode=dict(start=[8], end=[11], datatype=str, description='ID code of this entry.'),
                                      chainID=dict(start=[13], end=[13], datatype=str, description='Chain identifier.'),
                                      seqBegin=dict(start=[15], end=[18], datatype=int, description='Initial sequence number of the PDBparser sequence segment, right justified.'),
                                      insertBegin=dict(start=[19], end=[19], datatype=str, description='Initial insertion code of the PDBparser sequence segment.'),
                                      seqEnd=dict(start=[21], end=[24], datatype=int, description='Ending sequence number of the PDBparser sequence segment, right justified.'),
                                      insertEnd=dict(start=[25], end=[25], datatype=str, description='Ending insertion code of the PDBparser sequence  segment.'),
                                      database=dict(start=[27], end=[32], datatype=str, description='Sequence database name.'),
                                      dbIdCode=dict(start=[48], end=[67], datatype=str, description='Sequence database identification code, left justified.')
                                      ))

        pdb_format.update(DBREF2=dict(idCode=dict(start=[8], end=[11], datatype=str, description='ID code of this entry.'),
                                      chainID=dict(start=[13], end=[13], datatype=str, description='Chain identifier.'),
                                      dbAccession=dict(start=[19], end=[40], datatype=str, description='Sequence database accession code, left justified.'),
                                      seqBegin=dict(start=[46], end=[55], datatype=int, description='nitial sequence number of the Database segment, right justified.'),
                                      seqEnd=dict(start=[58], end=[67], datatype=int, description='Ending sequence number of the Database segment, right justified.')
                                      ))

        pdb_format.update(SEQRES=dict(
                                      serNum=dict(start=[8], end=[10], datatype=int, description='Serial number of the SEQRES record for  the current  chain. Starts at 1 and increments by one  each line. Reset to 1 for each chain.'),
                                      chainID=dict(start=[12], end=[12], datatype=str, description='Chain identifier. This may be any single legal  character, including a blank which is is  used if there is only one chain.'),
                                      numRes=dict(start=[14], end=[17], datatype=int, description='Number of residues in the chain. This  value is repeated on every record.'),
                                      resName=dict(start=[20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68], end=[22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70], datatype=str, description='Residue name.'),
                                      # resName=dict(start=24], end=[26, datatype=str, description='Residue name.'),
                                      # resName=dict(start=28, end=30, datatype=str, description='Residue name.'),
                                      # resName=dict(start=32, end=34, datatype=str, description='Residue name.'),
                                      # resName=dict(start=36, end=38, datatype=str, description='Residue name.'),
                                      # resName=dict(start=40, end=42, datatype=str, description='Residue name.'),
                                      # resName=dict(start=44, end=46, datatype=str, description='Residue name.'),
                                      # resName=dict(start=48, end=50, datatype=str, description='Residue name.'),
                                      # resName=dict(start=52, end=54, datatype=str, description='Residue name.'),
                                      # resName=dict(start=56, end=58, datatype=str, description='Residue name.'),
                                      # resName=dict(start=60, end=62, datatype=str, description='Residue name.'),
                                      # resName=dict(start=64, end=66, datatype=str, description='Residue name.'),
                                      # resName=dict(start=68, end=70, datatype=str, description='Residue name.')
                                      ))

        pdb_format.update(HET   =dict(hetID=dict(start=[8], end=[10], datatype=str, description='Het identifier, right-justified.'),
                                      chainID=dict(start=[13], end=[13], datatype=str, description='Chain  identifier.'),
                                      seqNum=dict(start=[14], end=[17], datatype=int, description='Sequence  number.'),
                                      iCode=dict(start=[18], end=[18], datatype=str, description='Insertion  code.'),
                                      numHetAtoms=dict(start=[21], end=[25], datatype=int, description='Number of HETATM records for the group present in the entry.'),
                                      text=dict(start=[31], end=[70], datatype=str, description='Text describing Het group.')
                                      ))

        pdb_format.update(HETNAM=dict(continuation=dict(start=[9], end=[10], datatype=str, description='Allows concatenation of multiple records.'),
                                      hetID=dict(start=[12], end=[14], datatype=str, description='Het identifier, right-justified.'),
                                      text=dict(start=[16], end=[70], datatype=str, description='Chemical name.')
                                      ))

        pdb_format.update(HETSYN=dict(continuation=dict(start=[9], end=[10], datatype=str, description='Allows concatenation of multiple records.'),
                                      hetID=dict(start=[12], end=[14], datatype=str, description='Het identifier, right-justified.'),
                                      hetSynonyms=dict(start=[16], end=[70], datatype=str, description='List of synonyms.')
                                      ))

        pdb_format.update(FORMUL=dict(compNum=dict(start=[9], end=[10], datatype=int, description='Component  number.'),
                                      hetID=dict(start=[13], end=[15], datatype=str, description='Het identifier.'),
                                      continuation=dict(start=[17], end=[18], datatype=str, description='Continuation number.'),
                                      asterisk=dict(start=[19], end=[19], datatype=str, description=' "*" for water.'),
                                      text=dict(start=[20], end=[70], datatype=str, description='Chemical formula.')
                                      ))

        pdb_format.update(HELIX =dict(serNum=dict(start=[8], end=[10], datatype=int, description='Serial number of the helix. This starts at 1  and increases incrementally.'),
                                      helixID=dict(start=[12], end=[14], datatype=str, description='Helix  identifier. In addition to a serial number, each helix is given an alphanumeric character helix identifier.'),
                                      initResName=dict(start=[16], end=[18], datatype=str, description='Name of the initial residue.'),
                                      initChainID=dict(start=[20], end=[20], datatype=str, description='Chain identifier for the chain containing this  helix.'),
                                      initSeqNum=dict(start=[22], end=[25], datatype=int, description='Sequence number of the initial residue.'),
                                      initICode=dict(start=[26], end=[26], datatype=str, description='Insertion code of the initial residue.'),
                                      endResName=dict(start=[28], end=[30], datatype=str, description='Name of the terminal residue of the helix.'),
                                      endChainID=dict(start=[32], end=[32], datatype=str, description='Chain identifier for the chain containing this  helix.'),
                                      endSeqNum=dict(start=[34], end=[37], datatype=int, description='Sequence number of the terminal residue.'),
                                      endICode=dict(start=[38], end=[38], datatype=str, description='Insertion code of the terminal residue.'),
                                      helixClass=dict(start=[39], end=[40], datatype=str, description='Helix class (see below).'),
                                      comment=dict(start=[41], end=[70], datatype=str, description='Comment about this helix.'),
                                      length=dict(start=[72], end=[76], datatype=int, description='Length of this helix.')
                                      ))

        pdb_format.update(SHEET=dict(strand=dict(start=[8], end=[10], datatype=int, description='Strand  number which starts at 1 for each strand within a sheet and increases by one.'),
                                     sheetID=dict(start=[12], end=[14], datatype=str, description='Sheet  identifier.'),
                                     numStrands=dict(start=[15], end=[16], datatype=int, description='Number  of strands in sheet.'),
                                     initResName=dict(start=[18], end=[20], datatype=str, description='Residue  name of initial residue.'),
                                     initChainID=dict(start=[22], end=[22], datatype=str, description='Chain identifier of initial residue in strand.'),
                                     initSeqNum=dict(start=[23], end=[26], datatype=int, description='Sequence number of initial residue in strand.'),
                                     initICode=dict(start=[27], end=[27], datatype=str, description=' Insertion code of initial residue in  strand.'),
                                     endResName=dict(start=[29], end=[31], datatype=str, description=' Residue name of terminal residue.'),
                                     endChainID=dict(start=[33], end=[33], datatype=str, description='Chain identifier of terminal residue.'),
                                     endSeqNum=dict(start=[34], end=[37], datatype=int, description=' Sequence number of terminal residue.'),
                                     endICode=dict(start=[38], end=[38], datatype=str, description='Insertion code of terminal residue.'),
                                     sense=dict(start=[39], end=[40], datatype=int, description='Sense of strand with respect to previous strand in the sheet. 0 if first strand, 1 if  parallel,and -1 if anti-parallel.'),
                                     curAtom=dict(start=[42], end=[45], datatype=str, description='Registration.  Atom name in current strand.'),
                                     curResName=dict(start=[46], end=[48], datatype=str, description='Registration.  Residue name in current strand'),
                                     curChainId=dict(start=[50], end=[50], datatype=str, description='Registration. Chain identifier in current strand.'),
                                     curResSeq=dict(start=[51], end=[54], datatype=str, description='Registration.  Residue sequence number in current strand.'),
                                     curICode=dict(start=[55], end=[55], datatype=str, description=' Registration. Insertion code in current strand.'),
                                     prevAtom=dict(start=[57], end=[60], datatype=str, description='Registration.  Atom name in previous strand.'),
                                     prevResName=dict(start=[61], end=[63], datatype=str, description='Registration.  Residue name in previous strand.'),
                                     prevChainId=dict(start=[65], end=[65], datatype=str, description='Registration.  Chain identifier in previous  strand.'),
                                     prevResSeq=dict(start=[66], end=[69], datatype=str, description=' Registration. Residue sequence number in previous strand.'),
                                     prevICode=dict(start=[70], end=[70], datatype=str, description='Registration.  Insertion code in previous strand.')
                                      ))

        pdb_format.update(SSBOND=dict(serNum=dict(start=[8], end=[10], datatype=int, description='Serial number.'),
                                    CYS1=dict(start=[12], end=[14], datatype=str, description='Residue name.'),
                                    chainID1=dict(start=[16], end=[16], datatype=str, description='Chain identifier.'),
                                    seqNum1=dict(start=[18], end=[21], datatype=int, description='Residue sequence number.'),
                                    icode1=dict(start=[22], end=[22], datatype=str, description='Insertion code.'),
                                    CYS2=dict(start=[26], end=[28], datatype=str, description='Residue name.'),
                                    chainID2=dict(start=[30], end=[30], datatype=str, description='Chain identifier.'),
                                    seqNum2=dict(start=[32], end=[35], datatype=int, description='Residue sequence number.'),
                                    icode2=dict(start=[36], end=[36], datatype=str, description='Insertion code.'),
                                    sym1=dict(start=[60], end=[65], datatype=str, description='Symmetry operator for residue 1.'),
                                    sym2=dict(start=[67], end=[72], datatype=str, description='Symmetry operator for residue 2.'),
                                    Length=dict(start=[74], end=[78], datatype=float, description='Disulfide bond distance.')
                                      ))

        pdb_format.update(LINK=dict(name1=dict(start=[13], end=[16], datatype=str, description='Atom name.'),
                                    altLoc1=dict(start=[17], end=[17], datatype=str, description='Alternate location indicator.'),
                                    resName1=dict(start=[18], end=[20], datatype=str, description='Residue  name.'),
                                    chainID1=dict(start=[22], end=[22], datatype=str, description='Chain identifier.'),
                                    resSeq1=dict(start=[23], end=[26], datatype=int, description='Residue sequence number.'),
                                    iCode1=dict(start=[27], end=[27], datatype=str, description='Insertion code.'),
                                    name2=dict(start=[43], end=[46], datatype=str, description='Atom name.'),
                                    altLoc2=dict(start=[47], end=[47], datatype=str, description='Alternate location indicator.'),
                                    resName2=dict(start=[48], end=[50], datatype=str, description='Residue name.'),
                                    chainID2=dict(start=[52], end=[52], datatype=str, description='Chain identifier.'),
                                    resSeq2=dict(start=[53], end=[56], datatype=int, description='Residue sequence number.'),
                                    iCode2=dict(start=[57], end=[57], datatype=str, description='Insertion code.'),
                                    sym1=dict(start=[60], end=[65], datatype=str, description='Symmetry operator for residue 1.'),
                                    sym2=dict(start=[67], end=[72], datatype=str, description='Symmetry operator for residue 2.'),
                                    Length=dict(start=[74], end=[78], datatype=float, description='Link distance.')
                                      ))

        pdb_format.update(CISPEP=dict(serNum=dict(start=[8], end=[10], datatype=int, description='Record serial number.'),
                                    pep1=dict(start=[12], end=[14], datatype=str, description='Residue name.'),
                                    chainID1=dict(start=[16], end=[16], datatype=str, description='Chain identifier.'),
                                    seqNum1=dict(start=[18], end=[21], datatype=int, description='Residue sequence number.'),
                                    iCode1=dict(start=[22], end=[22], datatype=str, description='Insertion code.'),
                                    pep2=dict(start=[26], end=[28], datatype=str, description='Residue name.'),
                                    chainID2=dict(start=[30], end=[30], datatype=str, description='Chain identifier.'),
                                    seqNum2=dict(start=[32], end=[35], datatype=int, description='Residue sequence number.'),
                                    iCode2=dict(start=[36], end=[36], datatype=str, description='Insertion code.'),
                                    modNum=dict(start=[44], end=[46], datatype=int, description='Identifies the specific model.'),
                                    measure=dict(start=[54], end=[59], datatype=float, description='Angle measurement in degrees.')
                                      ))

        pdb_format.update(SITE=dict(seqNum=dict(start=[8], end=[10], datatype=int, description='Sequence number.'),
                                    siteID=dict(start=[12], end=[14], datatype=str, description='Site name.'),
                                    numRes=dict(start=[16], end=[17], datatype=int, description='Number of residues that compose the site.'),
                                    resName1=dict(start=[19], end=[21], datatype=str, description='Residue name for first residue that creates the site.'),
                                    chainID1=dict(start=[23], end=[23], datatype=str, description='Chain identifier for first residue of site.'),
                                    seq1=dict(start=[24], end=[27], datatype=str, description='Residue sequence number for first residue of the  site.'),
                                    iCode1=dict(start=[28], end=[28], datatype=str, description='Insertion code for first residue of the site.'),
                                    resName2=dict(start=[30], end=[32], datatype=str, description='Residue name for second residue that creates the site.'),
                                    chainID2=dict(start=[34], end=[34], datatype=str, description='Chain identifier for second residue of the  site.'),
                                    seq2=dict(start=[35], end=[38], datatype=str, description='Residue sequence number for second residue of the site.'),
                                    iCode2=dict(start=[39], end=[39], datatype=str, description='Insertion code for second residue of the  site.'),
                                    resName3=dict(start=[41], end=[43], datatype=str, description='Residue name for third residue that  creates  the site.'),
                                    chainID3=dict(start=[45], end=[45], datatype=str, description='Chain identifier for third residue of the site.'),
                                    seq3=dict(start=[46], end=[49], datatype=str, description='Residue sequence number for third residue of the site.'),
                                    iCode3=dict(start=[50], end=[50], datatype=str, description='Insertion code for third residue of the site.'),
                                    resName4=dict(start=[52], end=[54], datatype=str, description='Residue name for fourth residue that creates  the site.'),
                                    chainID4=dict(start=[56], end=[56], datatype=str, description='Chain identifier for fourth residue of the site.'),
                                    seq4=dict(start=[57], end=[60], datatype=str, description='Residue sequence number for fourth residue of the site.'),
                                    iCode4=dict(start=[61], end=[16], datatype=str, description='Insertion code for fourth residue of the site.')
                                      ))

        pdb_format.update(CRYST1=dict(a=dict(start=[7], end=[15], datatype=float, description='Record serial number.'),
                                    b=dict(start=[16], end=[24], datatype=float, description='Residue name.'),
                                    c=dict(start=[25], end=[33], datatype=float, description='Chain identifier.'),
                                    alpha=dict(start=[34], end=[40], datatype=float, description='Residue sequence number.'),
                                    beta=dict(start=[41], end=[47], datatype=float, description='Insertion code.'),
                                    gamma=dict(start=[48], end=[54], datatype=float, description='Residue name.'),
                                    sGroup=dict(start=[56], end=[66], datatype=str, description='Chain identifier.'),
                                    z=dict(start=[67], end=[70], datatype=int, description='Residue sequence number.')
                                      ))

        # pdb_format.update(ORIGXn=dict(O_n1=dict(start=[11], end=[20], datatype=float, description='On1'),
        #                             O_n2=dict(start=[21], end=[30], datatype=float, description='On2'),
        #                             O_n3=dict(start=[31], end=[40], datatype=float, description='On3'),
        #                             T_n=dict(start=[46], end=[55], datatype=float, description='Tn')
        #                               ))


        pdb_format.update(ORIGX1=dict(O_11=dict(start=[11], end=[20], datatype=float, description='On1'),
                                    O_12=dict(start=[21], end=[30], datatype=float, description='On2'),
                                    O_13=dict(start=[31], end=[40], datatype=float, description='On3'),
                                    T_1=dict(start=[46], end=[55], datatype=float, description='Tn')
                                      ))

        pdb_format.update(ORIGX2=dict(O_21=dict(start=[11], end=[20], datatype=float, description='On1'),
                                    O_22=dict(start=[21], end=[30], datatype=float, description='On2'),
                                    O_23=dict(start=[31], end=[40], datatype=float, description='On3'),
                                    T_2=dict(start=[46], end=[55], datatype=float, description='Tn')
                                      ))

        pdb_format.update(ORIGX3=dict(O_31=dict(start=[11], end=[20], datatype=float, description='On1'),
                                    O_32=dict(start=[21], end=[30], datatype=float, description='On2'),
                                    O_33=dict(start=[31], end=[40], datatype=float, description='On3'),
                                    T_3=dict(start=[46], end=[55], datatype=float, description='Tn')
                                      ))

        # pdb_format.update(SCALEn=dict(S_n1=dict(start=[11], end=[20], datatype=float, description='Sn1'),
        #                             S_n2=dict(start=[21], end=[30], datatype=float, description='Sn2'),
        #                             S_n3=dict(start=[31], end=[40], datatype=float, description='Sn3'),
        #                             U_n=dict(start=[46], end=[55], datatype=float, description='Un')
        #                               ))

        pdb_format.update(SCALE1=dict(S_11=dict(start=[11], end=[20], datatype=float, description='Sn1'),
                                    S_12=dict(start=[21], end=[30], datatype=float, description='Sn2'),
                                    S_13=dict(start=[31], end=[40], datatype=float, description='Sn3'),
                                    U_1=dict(start=[46], end=[55], datatype=float, description='Un')
                                      ))

        pdb_format.update(SCALE2=dict(S_21=dict(start=[11], end=[20], datatype=float, description='Sn1'),
                                    S_22=dict(start=[21], end=[30], datatype=float, description='Sn2'),
                                    S_23=dict(start=[31], end=[40], datatype=float, description='Sn3'),
                                    U_2=dict(start=[46], end=[55], datatype=float, description='Un')
                                      ))

        pdb_format.update(SCALE3=dict(S_31=dict(start=[11], end=[20], datatype=float, description='Sn1'),
                                    S_32=dict(start=[21], end=[30], datatype=float, description='Sn2'),
                                    S_33=dict(start=[31], end=[40], datatype=float, description='Sn3'),
                                    U_3=dict(start=[46], end=[55], datatype=float, description='Un')
                                      ))

        # pdb_format.update(MTRIXn=dict(serial=dict(start=[8], end=[10], datatype=int, description='Sn1'),
        #                             M_n1=dict(start=[11], end=[20], datatype=float, description='Sn2'),
        #                             M_n2=dict(start=[21], end=[30], datatype=float, description='Sn2'),
        #                             M_n3=dict(start=[31], end=[40], datatype=float, description='Sn3'),
        #                             V_n=dict(start=[46], end=[55], datatype=float, description='Un')
        #                               ))

        pdb_format.update(MTRIX1=dict(serial=dict(start=[8], end=[10], datatype=int, description='Sn1'),
                                    M_11=dict(start=[11], end=[20], datatype=float, description='Sn2'),
                                    M_12=dict(start=[21], end=[30], datatype=float, description='Sn2'),
                                    M_13=dict(start=[31], end=[40], datatype=float, description='Sn3'),
                                    V_1=dict(start=[46], end=[55], datatype=float, description='Un')
                                      ))

        pdb_format.update(MTRIX2=dict(serial=dict(start=[8], end=[10], datatype=int, description='Sn1'),
                                    M_21=dict(start=[11], end=[20], datatype=float, description='Sn2'),
                                    M_22=dict(start=[21], end=[30], datatype=float, description='Sn2'),
                                    M_23=dict(start=[31], end=[40], datatype=float, description='Sn3'),
                                    V_2=dict(start=[46], end=[55], datatype=float, description='Un')
                                      ))

        pdb_format.update(MTRIX3=dict(serial=dict(start=[8], end=[10], datatype=int, description='Sn1'),
                                    M_31=dict(start=[11], end=[20], datatype=float, description='Sn2'),
                                    M_32=dict(start=[21], end=[30], datatype=float, description='Sn2'),
                                    M_33=dict(start=[31], end=[40], datatype=float, description='Sn3'),
                                    V_3=dict(start=[46], end=[55], datatype=float, description='Un')
                                      ))

        pdb_format.update(MODEL =dict(serial=dict(start=[11], end=[14], datatype=int, description='Model serial number.')
                                      ))

        pdb_format.update(ANISOU=dict(serial=dict(start=[7], end=[11], datatype=int, description='Atom serial number.'),
                                    name=dict(start=[13], end=[16], datatype=str, description='Atom name.'),
                                    altLoc=dict(start=[17], end=[17], datatype=str, description='Alternate location indicator'),
                                    resName=dict(start=[18], end=[20], datatype=str, description='Residue name.'),
                                    chainID=dict(start=[22], end=[22], datatype=str, description='Chain identifier.'),
                                    resSeq=dict(start=[23], end=[26], datatype=int, description='Residue sequence number.'),
                                    iCode=dict(start=[27], end=[27], datatype=str, description='Insertion code.'),
                                    U_00=dict(start=[29], end=[35], datatype=int, description='U(1,1)'),
                                    U_11=dict(start=[36], end=[42], datatype=int, description='U(2,2)'),
                                    U_22=dict(start=[43], end=[49], datatype=int, description='U(3,3)'),
                                    U_01=dict(start=[50], end=[56], datatype=int, description='U(1,2)'),
                                    U_02=dict(start=[57], end=[63], datatype=int, description='U(1,3)'),
                                    U_12=dict(start=[64], end=[70], datatype=int, description='U(2,3)'),
                                    element=dict(start=[77], end=[78], datatype=str, description='Element symbol, right-justified.'),
                                    charge=dict(start=[79], end=[80], datatype=str, description='Charge on the atom.')
                                      ))

        pdb_format.update(HETATM=dict(serial=dict(start=[7], end=[11], datatype=int, description='Atom serial number.'),
                                    name=dict(start=[13], end=[16], datatype=str, description='Atom name.'),
                                    altLoc=dict(start=[17], end=[17], datatype=str, description='Alternate location indicator'),
                                    resName=dict(start=[18], end=[20], datatype=str, description='Residue name.'),
                                    chainID=dict(start=[22], end=[22], datatype=str, description='Chain identifier.'),
                                    resSeq=dict(start=[23], end=[26], datatype=int, description='Residue sequence number.'),
                                    iCode=dict(start=[27], end=[27], datatype=str, description='Insertion code.'),
                                    x=dict(start=[31], end=[38], datatype=float, description='Orthogonal coordinates for X.'),
                                    y=dict(start=[39], end=[46], datatype=float, description='Orthogonal coordinates for Y.'),
                                    z=dict(start=[47], end=[54], datatype=float, description='Orthogonal coordinates for Z.'),
                                    occupancy=dict(start=[55], end=[60], datatype=float, description='Occupancy.'),
                                    tempFactor=dict(start=[61], end=[66], datatype=float, description='Temperature factor.'),
                                    element=dict(start=[77], end=[78], datatype=str, description='Element symbol; right-justified.'),
                                    charge=dict(start=[79], end=[80], datatype=str, description='Charge on the atom.')
                                      ))

        pdb_format.update(ATOM=dict(serial=dict(start=[7], end=[11], datatype=int, description='Atom  serial number.'),
                                    name=dict(start=[13], end=[16], datatype=str, description='Atom name.'),
                                    altLoc=dict(start=[17], end=[17], datatype=str, description='Alternate location indicator.'),
                                    resName=dict(start=[18], end=[20], datatype=str, description='Residue name.'),
                                    chainID=dict(start=[22], end=[22], datatype=str, description='Chain identifier.'),
                                    resSeq=dict(start=[23], end=[26], datatype=int, description='Residue sequence number.'),
                                    iCode=dict(start=[27], end=[27], datatype=str, description='Code for insertion of residues.'),
                                    x=dict(start=[31], end=[38], datatype=float, description='Orthogonal coordinates for X in Angstroms.'),
                                    y=dict(start=[39], end=[46], datatype=float, description='Orthogonal coordinates for Y in Angstroms.'),
                                    z=dict(start=[47], end=[54], datatype=float, description='Orthogonal coordinates for Z in Angstroms.'),
                                    occupancy=dict(start=[55], end=[60], datatype=float, description='Occupancy.'),
                                    tempFactor=dict(start=[61], end=[66], datatype=float, description='Temperature  factor.'),
                                    element=dict(start=[77], end=[78], datatype=str, description='Element symbol, right-justified.'),
                                    charge=dict(start=[79], end=[80], datatype=str, description='Charge  on the atom.')))

        pdb_format.update(TER=dict(serial=dict(start=[7], end=[11], datatype=int, description='Atom serial number.'),
                                    resName=dict(start=[18], end=[20], datatype=str, description='Residue name.'),
                                    chainID=dict(start=[22], end=[22], datatype=str, description='Chain identifier.'),
                                    resSeq=dict(start=[23], end=[26], datatype=int, description='Residue sequence number.'),
                                    iCode=dict(start=[27], end=[27], datatype=str, description='Insertion code.')
                                      ))

        pdb_format.update(ENDMDL=dict())

        pdb_format.update(CONECT=dict(serial=dict(start=[7, 12, 17, 22, 27], end=[11, 16, 21, 26, 31], datatype=str, description='Insertion code.'),
                                    # serial= dict(start=[12], end=[16], datatype=int, description='Chain identifier.'),
                                    # serial= dict(start=[17], end=[21], datatype=int, description='Chain identifier.'),
                                    # serial= dict(start=[22], end=[26], datatype=int, description='Chain identifier.'),
                                    # serial= dict(start=[27], end=[31], datatype=int, description='Chain identifier.')
                                      ))

        pdb_format.update(MASTER=dict(numRemark=dict(start=[11], end=[15], datatype=int, description='Atom serial number.'),
                                    _0=dict(start=[16], end=[20], datatype=int, description='Residue name.'),
                                    numHet=dict(start=[21], end=[25], datatype=int, description='Residue name.'),
                                    numHelix=dict(start=[26], end=[30], datatype=int, description='Chain identifier.'),
                                    numSheet=dict(start=[31], end=[35], datatype=int, description='Chain identifier.'),
                                    numTurn=dict(start=[36], end=[40], datatype=int, description='Chain identifier.'),
                                    numSite=dict(start=[41], end=[45], datatype=int, description='Chain identifier.'),
                                    numXform=dict(start=[46], end=[50], datatype=int, description='Residue sequence number.'),
                                    numCoord=dict(start=[51], end=[55], datatype=int, description='Residue sequence number.'),
                                    numTer=dict(start=[56], end=[60], datatype=int, description='Residue sequence number.'),
                                    numConect=dict(start=[61], end=[65], datatype=int, description='Residue sequence number.'),
                                    numSeq=dict(start=[66], end=[70], datatype=int, description='Insertion code.')
                                      ))

        pdb_format.update(END=dict())

        return pdb_format
