import difflib
import traceback

from lxml import etree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def extract_text_and_tags(xml_string,show_errors=True):
    """Extract text and tag content from an XML string. Return None on failure."""
    try:
        xslt_content = xml_string.encode('ascii')
        # root = etree.XML(xslt_content)
        root = etree.fromstring(xslt_content)
        texts = []
        tags = []
        # print(root)
        for elem in root.iter():
            tags.append(str(elem.tag))
            if elem.text and elem.text.strip():
                texts.append(str(elem.text.strip()))

        return ' '.join(texts), ' '.join(tags)
    except Exception:
        if show_errors:
            traceback.print_exc()
        return None


def text_similarity(text1, text2):
    """Compute cosine similarity between two texts."""
    vectorizer = TfidfVectorizer().fit([text1, text2])
    tfidf = vectorizer.transform([text1, text2])
    return cosine_similarity(tfidf[0], tfidf[1])[0, 0]


def tag_similarity(tags1, tags2):
    """Compute normalized similarity using SequenceMatcher."""
    return difflib.SequenceMatcher(None, tags1, tags2).ratio()


def combined_similarity(xml1, xml2, alpha=0.5,show_errors=True):
    """Combine text and tag similarity with a weighted alpha parameter."""
    result1 = extract_text_and_tags(xml1,show_errors)
    result2 = extract_text_and_tags(xml2,show_errors)

    if result1 is None or result2 is None:
        # Fall back to raw text comparison, no tag similarity
        sim_text = text_similarity(xml1, xml2)
        sim_tags = 0.0
    else:
        text1, tags1 = result1
        text2, tags2 = result2
        sim_text = text_similarity(text1, text2)
        sim_tags = tag_similarity(tags1, tags2)

    combined_score = alpha * sim_text + (1 - alpha) * sim_tags
    return {
        'text_similarity': sim_text,
        'tag_similarity': sim_tags,
        'combined_similarity': combined_score
    }


# Example usage:
if __name__ == "__main__":
    xml1 = """<root><title>Hello</title><body>This is a test.</body></root>"""
    xml2 = """<root><title>Hello/title><body>This is an exam.</body></root>"""  # malformed

    result = combined_similarity(xml1, xml2, alpha=0.7)
    print(result)
