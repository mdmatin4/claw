from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
import re
import shutil, gc

##################
def split_text_v1(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        # move start forward with overlap
        start += chunk_size - overlap

    return chunks
####################
## Honors sentence boundary

def split_text_v2(text, chunk_size=500, overlap=100):
    sentences = re.split(r'(?<=[.!?]) +', text)
    
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())

            # create overlap from last part
            overlap_text = current_chunk[-overlap:] if overlap > 0 else ""
            current_chunk = overlap_text + " " + sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

############################
## Main
folders = [
    r"C:\Users\munsh\Desktop\10102020\NCI\ITServe\DocAI\web scraping\LLM\Taiho",
    r"C:\Users\munsh\Desktop\10102020\NCI\ITServe\DocAI\web scraping\LLM\CV",
]

db_path = Path("./vector_db/file_index")
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./vector_db/file_index")
# collection = client.get_or_create_collection("local_files")
collection_name = "local_files"

existing = [c.name for c in client.list_collections()]

if collection_name in existing:
    choice = input(
        "Collection already exists. Type 'load' to use it or 'recreate' to delete and rebuild: "
    ).strip().lower()

    if choice == "recreate":
        client.delete_collection(collection_name)
        collection = None
        client = None
        shutil.rmtree(db_path)
        gc.collect()
        time.sleep(1)
        print("Deleted full vector DB folder")
        collection = client.create_collection(collection_name)
        print("Collection recreated, now going to create indexes...")
        run_indexing = True

    else:  # default = load
        collection = client.get_collection(collection_name)
        print("Existing Collection loaded with all indexes!")
        run_indexing = False

else:
    collection = client.create_collection(collection_name)
    print("Collection created")
    run_indexing = True


# Only run indexing if needed
if run_indexing:
    print("Running indexing...\n")
    for folder in folders:
        for file in Path(folder).rglob("*"):
            if file.suffix.lower() in [".txt", ".pdf", ".docx", ".xlsx", ".csv"]:
                ## extract text here
                # text = extract_text(file)
            
                text="""Dr. Mohammad Husain is a Professor and the Inaugural Director of the PolySec Cyber Lab, a Center for Cyber 
Security and Forensics Education, Research, and Outreach in the Department of Computer Science at the California 
State Polytechnic University, Pomona (Cal Poly Pomona), CA. He has over five years’ experience in leadership, 
development, and administration of Computer Science and Cyber Security academic and co-curricular programs at a 
comprehensive university. He has significant leadership experience in extramural funding, graduate program, 
financial and strategic planning, assessment and accreditation, faculty governance, and building academic vision in 
diverse academic environments. His academic vision and leadership skills have resulted in state and national-level 
programs and diverse academic settings across multiple institutions.
Dr. Husain’s administrative leadership includes the Program Directorship of the Cal-Bridge Computer Science (CS) 
program, a California State University (CSU)-University of California (UC) Ph.D. pathway program for Computer 
Science and Engineering students from underrepresented minority communities. Dr. Husain leads the steering 
committee comprised of six CSUs, six UCs, and one community college. The program was launched in Fall 2020 and 
recruited 18 CSU students successfully. He is also the founding organizing chair of the SFSCon, a national 
cybersecurity training workshop for the CyberCorps Scholarship for Service (SFS) students. In Fall 2021, around 141 
CyberCorps SFS students (BS/MS/Ph.D.) from 42 different US universities participated in this 2-day training.
At Cal Poly Pomona, he is the Principal Investigator (PI) of nine externally funded grants and co-PI of two externally 
funded grants totaling over $4.1M. As a part of the NSF CyberCorps SFS grant, he has trained students in Cyber 
Security and placed them at the NSA, DHS, NASA, MITRE, Sandia lab, Aerospace Corp., MIT Lincoln Lab, US 
Airforce, US Army, and US Navy since 2015. As a part of the ONR grant, he has trained 45Navy, Marine, and Army 
ROTC cadets from the following schools: USC, UCLA, UCSD, USD, SDSU, Point Loma, CSU San Marcos, and CPP. 
As the administrative lead of these grants, Dr. Husain has also managed over 30+ faculty and staff since 2015.
As an academic leader at Cal Poly Pomona, Dr. Husain has led multiple university IT initiatives: Cyber Security 
Cluster Hiring (2017), Big Data Interdisciplinary Scholarly Community (2018), High-Performance Computing Lab 
(2017), Virtual Reality Lab (2017), and Malware Analysis & Security Operations Center (2018). Additionally, he has 
served on the Executive Committee (2019) and Academic Affairs committee of the CPP Academic Senate (2015-19), 
as well as on the University Academic Master Plan Committee (2016). Dr. Husain also served as the founding faculty 
advisor of ACM-W chapter CPP SheCodes through support from the National Center of Women in IT (2016). This 
chapter's strategic recruitment and retention program have resulted in a 21% increase in female applicants to the CS 
program in 2020. He has held leadership positions at the department level on the following committees: Retention 
Tenure and Promotion (RTP), Laboratory, Scholarship, and Graduate Committee. He is the CS Graduate committee's 
current chair and has been in charge of the MS in Computer Science program since 2015.
Dr. Husain has more than 30 peer-reviewed publications and over 50 technical publications, including two coauthored book chapters in the domain of Cyber Security and Forensics. His research on Brainwave authentication 
was assigned a US patent (USPTO 10,198,566) and covered by Time magazine and PC magazine. Dr. Husain is a 
member of various professional organizations: IEEE, ACM, and Usenix (2015-present). He served as the publicity 
chair of the International Conference on Security Knowledge Management (2019) and a member of the technical 
program committee of IEEE WCNC (2014-present), WASA (2013-14), ICCVE (2013-14), and WTS (2013). He has 
served as a reviewer for National Science Foundation (NSF), IEEE Transactions on Secure and Distributed 
Computing (2013), IEEE Transactions on Cloud Computing (2013), and Elsevier Ad hoc Networks Journal (2012).
He earned an MS and Ph.D. in Computer Science and Engineering from the State University of New York at Buffalo 
in 2012. Dr. Husain is the recipient of the 2016 College of Science Distinguished Teaching Award and was awarded 
early promotion and tenure in 2016. Cal Poly Pomona also selected Dr. Husain as its 2020 Faculty Learning 
Community for Leadership pipeline development cohort.
"""

                chunks = split_text_v2(text)


                for i, chunk in enumerate(chunks):
                    embedding = model.encode(chunk).tolist()

                    collection.add(
                        ids=[f"{file}-{i}"],
                        embeddings=[embedding],
                        documents=[chunk],
                        metadatas=[{
                            "file_path": str(file),
                            "file_name": file.name
                        }]
                    )
###########################
## Test the code here
while True:
    query = input("Ask a question (type 'exit' to quit): ").strip()

    if query.lower() == "exit":
        print("Exiting...")
        break

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )
    print(results)
    print("\nHere is the answer...\n")
    for doc in results["documents"][0]:
        print(doc)
        print("\n" + "-"*50 + "\n")
