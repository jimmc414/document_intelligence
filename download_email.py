import os
import base64
import re
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Downloaded credentials.json file path
cred_file = 'credentials.json'

# Email search criteria
to_criteria = '*'
from_criteria = 'jim.mcmillan@indexai.io'
subject_criteria = '*'
start_date = '2023/05/09'
end_date = '2023/06/01'

# Output directory
output_directory = 'C:/python/autoindex/txt_output/'

def authenticate():
    scopes = ['https://www.googleapis.com/auth/gmail.readonly']
    flow = InstalledAppFlow.from_client_secrets_file(cred_file, scopes)
    creds = flow.run_local_server(port=0)
    return creds

def save_email_to_file(content, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

def process_headers(headers):
    headers_processed = {}
    for header in headers:
        if header['name'] in ['To', 'From', 'Subject', 'Date']:
            headers_processed[header['name']] = header['value']
    return headers_processed

def create_filename(headers, output_directory):
    to_value = re.sub(r'\W+', '_', headers['To'])
    from_value = re.sub(r'\W+', '_', headers['From'])
    subject_value = re.sub(r'\W+', '_', headers['Subject'])
    date_value = re.sub(r'\W+', '_', headers['Date'])
    filename = f"{to_value}-{from_value}-{subject_value}-{date_value}.txt"
    full_path = os.path.join(output_directory, filename)
    return full_path

def get_text_plain_from_payload(payload):
    if 'mimeType' in payload and payload['mimeType'] == 'text/plain' and 'data' in payload['body']:
        return base64.urlsafe_b64decode(payload['body']['data'].encode('ASCII')).decode()
    if 'parts' in payload:
        for part in payload['parts']:
            text = get_text_plain_from_payload(part)
            if text:
                return text
    return None

def main():
    creds = authenticate()
    gmail = build('gmail', 'v1', credentials=creds)

    # Create the email query using provided criteria
    query_parts = []
    if to_criteria and to_criteria != '*':
        query_parts.append(f'to:{to_criteria}')
    if from_criteria and from_criteria != '*':
        query_parts.append(f'from:{from_criteria}')
    if subject_criteria and subject_criteria != '*':
        query_parts.append(f'subject:{subject_criteria}')
    if start_date and start_date != '*':
        query_parts.append(f'after:{start_date}')
    if end_date and end_date != '*':
        query_parts.append(f'before:{end_date}')
    
    query = ' '.join(query_parts)

    # Execute the search
    results = gmail.users().messages().list(userId='me', q=query).execute()
    messages = results.get('messages', [])

    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if not messages:
        print('No emails found.')
    else:
        print('Emails found:')
        for message in messages:
            # Read the email
            msg_data = gmail.users().messages().get(userId='me', id=message['id'], format='full').execute()

            # Extract headers and create output filename
            headers = process_headers(msg_data['payload']['headers'])
            filename = create_filename(headers, output_directory)

            # Extract the email body as a plain text
            msg_raw = get_text_plain_from_payload(msg_data['payload'])
            if msg_raw is None:
                print(f"Error: Couldn't retrieve plain text content for email - {filename}")
                continue
            
            save_email_to_file(msg_raw, filename)
            print(f'Saved email to: {filename}')

if __name__ == '__main__':
    main()