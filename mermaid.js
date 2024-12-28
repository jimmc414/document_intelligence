graph TB
    subgraph Input
        PDF[PDF Files]:::input
        Email[Email Attachments]:::input
        Audio[Audio Files]:::input
        FMS[File Management System]:::input
        click FMS "https://github.com/jimmc414/document_intelligence/blob/main/manage_files.py"
    end

    subgraph Processing
        TE[Text Extraction Services]:::processing
        
        PDF_P[PDF Processing]:::processing
        click PDF_P "https://github.com/jimmc414/document_intelligence/blob/main/extract_text_from_pdf.py"
        
        OCR[OCR Processing]:::processing
        click OCR "https://github.com/jimmc414/document_intelligence/blob/main/optical_character_recognition.py"
        
        Audio_P[Audio Processing]:::processing
        click Audio_P "https://github.com/jimmc414/document_intelligence/blob/main/extract_text_from_audio.py"
        
        Email_P[Email Processing]:::processing
        click Email_P "https://github.com/jimmc414/document_intelligence/blob/main/dl_email.py"
    end

    subgraph Analysis
        NER[Named Entity Recognition]:::analysis
        click NER "https://github.com/jimmc414/document_intelligence/blob/main/extract_named_entities.py"
        
        KV[Key-Value Extraction]:::analysis
        click KV "https://github.com/jimmc414/document_intelligence/blob/main/extract_key_value_pairs.py"
        
        SA[Sentiment Analysis]:::analysis
        click SA "https://github.com/jimmc414/document_intelligence/blob/main/sentiment_analysis.py"
        
        TS[Text Summarization]:::analysis
        click TS "https://github.com/jimmc414/document_intelligence/blob/main/summarize_text.py"
        
        TM[Topic Modeling]:::analysis
        click TM "https://github.com/jimmc414/document_intelligence/blob/main/create_topic_model.py"
    end

    subgraph Classification
        DC[Document Classification]:::classification
        click DC "https://github.com/jimmc414/document_intelligence/blob/main/document_classification.py"
        
        CS[Clustering Services]:::classification
        click CS "https://github.com/jimmc414/document_intelligence/blob/main/cluster_documents.py"
        
        SimA[Similarity Analysis]:::classification
        click SimA "https://github.com/jimmc414/document_intelligence/blob/main/document_similarity.py"
    end

    subgraph Integration
        ESI[External System Integration]:::integration
        AAS[Account Assignment System]:::integration
    end

    %% Connections
    PDF & Email & Audio --> FMS
    FMS --> TE
    TE --> PDF_P & OCR & Audio_P & Email_P
    PDF_P & OCR & Audio_P & Email_P --> NER & KV & SA & TS & TM
    NER & KV & SA & TS & TM --> DC
    DC --> CS & SimA
    CS & SimA --> ESI
    ESI --> AAS

    %% Styles
    classDef input fill:#a8d1f0
    classDef processing fill:#90EE90
    classDef analysis fill:#FFB347
    classDef classification fill:#DDA0DD
    classDef integration fill:#D3D3D3

    %% Legend
    subgraph Legend
        L1[Input Components]:::input
        L2[Processing Services]:::processing
        L3[Analysis Services]:::analysis
        L4[Classification Services]:::classification
        L5[Integration Components]:::integration
    end
