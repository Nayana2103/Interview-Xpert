import PyPDF2
import re
import string
import pandas as pd

def analyze_pdf_keywords(file_name):
    try:
        pdfFileObj = open("."+file_name, 'rb')
        pdfReader = PyPDF2.PdfReader(pdfFileObj)
        num_pages = len(pdfReader.pages)
        
        text = ""
        for count in range(num_pages):
            text += pdfReader.pages[count].extract_text() or ""
        
        pdfFileObj.close()
        
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))

        terms = {
            'Quality/Six Sigma': ['black belt', 'capability analysis', 'control charts', 'doe', 'dmaic', 'fishbone',
                                  'gage r&r', 'green belt', 'ishikawa', 'iso', 'kaizen', 'kpi', 'lean', 'metrics',
                                  'pdsa', 'performance improvement', 'process improvement', 'quality',
                                  'quality circles', 'quality tools', 'root cause', 'six sigma',
                                  'stability analysis', 'statistical analysis', 'tqm'],
            'Operations management': ['automation', 'bottleneck', 'constraints', 'cycle time', 'efficiency', 'fmea',
                                      'machinery', 'maintenance', 'manufacture', 'line balancing', 'oee', 'operations',
                                      'operations research', 'optimization', 'overall equipment effectiveness',
                                      'pfmea', 'process', 'process mapping', 'production', 'resources', 'safety',
                                      'stoppage', 'value stream mapping', 'utilization'],
            'Supply chain': ['abc analysis', 'apics', 'customer', 'customs', 'delivery', 'distribution', 'eoq', 'epq',
                             'fleet', 'forecast', 'inventory', 'logistic', 'materials', 'outsourcing', 'procurement',
                             'reorder point', 'rout', 'safety stock', 'scheduling', 'shipping', 'stock', 'suppliers',
                             'third party logistics', 'transport', 'transportation', 'traffic', 'supply chain',
                             'vendor', 'warehouse', 'wip', 'work in progress'],
            'Project management': ['administration', 'agile', 'budget', 'cost', 'direction', 'feasibility analysis',
                                   'finance', 'kanban', 'leader', 'leadership', 'management', 'milestones', 'planning',
                                   'pmi', 'pmp', 'problem', 'project', 'risk', 'schedule', 'scrum', 'stakeholders'],
            'Data analytics': ['analytics', 'api', 'aws', 'big data', 'business intelligence', 'clustering', 'code',
                               'coding', 'data', 'database', 'data mining', 'data science', 'deep learning', 'hadoop',
                               'hypothesis test', 'iot', 'internet', 'machine learning', 'modeling', 'nosql', 'nlp',
                               'predictive', 'programming', 'python', 'r', 'sql', 'tableau', 'text mining',
                               'visualization'],
            'Healthcare': ['adverse events', 'care', 'clinic', 'cphq', 'ergonomics', 'healthcare',
                           'health care', 'health', 'hospital', 'human factors', 'medical', 'near misses',
                           'patient', 'reporting system']
        }
        
        scores = {area: sum(1 for word in terms[area] if word in text) for area in terms.keys()}
        print("Scores:",scores)

        # summary = str()

        # summary = "Quality/Six Sigma : " + str(scores['Quality/Six Sigma']) + "\n" \
        #           "Operations management : " + str(scores['Operations management']) + "\n" \
        #           "Supply chain : " + str(scores['Supply chain']) + "\n" \
        #           "Project management : " + str(scores['Project management']) + "\n" \
        #           "Data analytics : " + str(scores['Data analytics']) + "\n" \
        #           "Healthcare : " + str(scores['Healthcare']) + "\n" \
        
        #summary = pd.DataFrame.from_dict(scores, orient='index', columns=['score']).sort_values(by='score', ascending=False)
        
        return scores
    
    except Exception as e:
        return f"Error: {str(e)}"

# Example Usage
# result = analyze_pdf_keywords("sample.pdf")
# print(result)
