import json
from bs4 import BeautifulSoup

def build_structured_table_json(
    table_img_path: str,
    table_caption: str,
    table_footnote: list,
    enhanced_caption: str,
    table_body_html: str
) -> str:
    """
    Converts an HTML <table>...</table> into structured JSON including metadata.
    """

    # --- Parse HTML table into rows & columns ---
    soup = BeautifulSoup(table_body_html, "html.parser")
    table = soup.find("table")

    rows = []
    columns = []

    for i, row in enumerate(table.find_all("tr")):
        cells = row.find_all(["th", "td"])
        cell_text = [c.get_text(strip=True) for c in cells]

        if i == 0:
            # Header row
            columns = cell_text
        else:
            # Data rows
            row_dict = {columns[j]: cell_text[j] for j in range(len(columns))}
            rows.append(row_dict)

    # --- Build JSON structure ---
    data = {
        "metadata": {
            "image_path": table_img_path,
            "caption": table_caption,
            "footnotes": table_footnote,
            "summary": enhanced_caption
        },
        "table": {
            "columns": columns,
            "rows": rows
        }
    }

    # Return pretty-printed JSON
    return json.dumps(data, indent=2)

table_html = """
<table><tr><td rowspan=1 colspan=1>Z2</td><td rowspan=1 colspan=1>ZB[mm]</td><td rowspan=1 colspan=1>0B[mm]</td><td rowspan=1 colspan=1>OTK[mm]</td><td rowspan=1 colspan=1>OKK[mm]</td><td rowspan=1 colspan=1>ON[mm]</td><td rowspan=1 colspan=1>L[mm]</td><td rowspan=1 colspan=1>OFM[mm]</td><td rowspan=1 colspan=1>Ws[mm]</td><td rowspan=1 colspan=1>G[g]</td><td rowspan=1 colspan=1>DM**[Ncm]</td><td rowspan=1 colspan=1>Art-Nr.</td></tr><tr><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>17.5</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>2,54</td><td rowspan=1 colspan=1>29,45</td><td rowspan=1 colspan=1>SH12512HF</td></tr><tr><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>16,25</td><td rowspan=1 colspan=1>18,75</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>2,92</td><td rowspan=1 colspan=1>31,91</td><td rowspan=1 colspan=1>SH12513HF</td></tr><tr><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>17.5</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>3.43</td><td rowspan=1 colspan=1>34.36</td><td rowspan=1 colspan=1>SH12514HF</td></tr><tr><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>18,75</td><td rowspan=1 colspan=1>21,25</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>3,79</td><td rowspan=1 colspan=1>36,82</td><td rowspan=1 colspan=1>SH12515HF</td></tr><tr><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>22.5</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>4,24</td><td rowspan=1 colspan=1>39,72</td><td rowspan=1 colspan=1>SH12516HF</td></tr><tr><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>21.25</td><td rowspan=1 colspan=1>23,75</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>4.5</td><td rowspan=1 colspan=1>41.72</td><td rowspan=1 colspan=1>SH12517HF</td></tr><tr><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>22,5</td><td rowspan=1 colspan=1>25</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>5,99</td><td rowspan=1 colspan=1>44,18</td><td rowspan=1 colspan=1>SH12518HF</td></tr><tr><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>23,75</td><td rowspan=1 colspan=1>26,25</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>6.62</td><td rowspan=1 colspan=1>46,63</td><td rowspan=1 colspan=1>SH12519HF</td></tr><tr><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>25</td><td rowspan=1 colspan=1>27.5</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>7,08</td><td rowspan=1 colspan=1>49,09</td><td rowspan=1 colspan=1>SH12520HF</td></tr><tr><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>26,25</td><td rowspan=1 colspan=1>28,75</td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>18,5</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>8,1</td><td rowspan=1 colspan=1>51,54</td><td rowspan=1 colspan=1>SH12521HF</td></tr><tr><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>27.5</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>18.5</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>9.14</td><td rowspan=1 colspan=1>54</td><td rowspan=1 colspan=1>SH12522HF</td></tr><tr><td rowspan=1 colspan=1>23</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>28,75</td><td rowspan=1 colspan=1>31,25</td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>18,5</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>9.75</td><td rowspan=1 colspan=1>56,45</td><td rowspan=1 colspan=1>SH12523HF</td></tr><tr><td rowspan=1 colspan=1>24</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>32.5</td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>10,45</td><td rowspan=1 colspan=1>58,9</td><td rowspan=1 colspan=1>SH12524HF</td></tr><tr><td rowspan=1 colspan=1>25</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>31.25</td><td rowspan=1 colspan=1>3375</td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>11,39</td><td rowspan=1 colspan=1>61,36</td><td rowspan=1 colspan=1>SH12525HF</td></tr><tr><td rowspan=1 colspan=1>26</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>32.5</td><td rowspan=1 colspan=1>35</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>23,5</td><td rowspan=1 colspan=1>5,5</td><td rowspan=1 colspan=1>12,52</td><td rowspan=1 colspan=1>63,81</td><td rowspan=1 colspan=1>SH12526HF</td></tr><tr><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>33,75</td><td rowspan=1 colspan=1>36,25</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>23.5</td><td rowspan=1 colspan=1>5.5</td><td rowspan=1 colspan=1>12.9</td><td rowspan=1 colspan=1>66,27</td><td rowspan=1 colspan=1>SH12527HF</td></tr><tr><td rowspan=1 colspan=1>28</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>35</td><td rowspan=1 colspan=1>37.5</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>23,5</td><td rowspan=1 colspan=1>5,5</td><td rowspan=1 colspan=1>13,81</td><td rowspan=1 colspan=1>68,72</td><td rowspan=1 colspan=1>SH12528HF</td></tr><tr><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>37.5</td><td rowspan=1 colspan=1>40</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>5,5</td><td rowspan=1 colspan=1>14,86</td><td rowspan=1 colspan=1>73.63</td><td rowspan=1 colspan=1>SH12530HF</td></tr><tr><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>40</td><td rowspan=1 colspan=1>42.5</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>5.5</td><td rowspan=1 colspan=1>17,04</td><td rowspan=1 colspan=1>78,54</td><td rowspan=1 colspan=1>SH12532HF</td></tr><tr><td rowspan=1 colspan=1>35</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>43,75</td><td rowspan=1 colspan=1>46,25</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>5,5</td><td rowspan=1 colspan=1>20,21</td><td rowspan=1 colspan=1>85,9</td><td rowspan=1 colspan=1>SH12535HF</td></tr><tr><td rowspan=1 colspan=1>36</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>45</td><td rowspan=1 colspan=1>47.5</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>36</td><td rowspan=1 colspan=1>5.5</td><td rowspan=1 colspan=1>18,21</td><td rowspan=1 colspan=1>88,36</td><td rowspan=1 colspan=1>SH12536HF</td></tr><tr><td rowspan=1 colspan=1>38</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>47.5</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>36</td><td rowspan=1 colspan=1>5.5</td><td rowspan=1 colspan=1>21,08</td><td rowspan=1 colspan=1>93,27</td><td rowspan=1 colspan=1>SH12538HF</td></tr><tr><td rowspan=1 colspan=1>40</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>52.5</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>36</td><td rowspan=1 colspan=1>5,5</td><td rowspan=1 colspan=1>23,07</td><td rowspan=1 colspan=1>98,17</td><td rowspan=1 colspan=1>SH12540HF</td></tr><tr><td rowspan=1 colspan=1>42</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>52.5</td><td rowspan=1 colspan=1>55</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>19</td><td rowspan=1 colspan=1>36</td><td rowspan=1 colspan=1>5.5</td><td rowspan=1 colspan=1>27</td><td rowspan=1 colspan=1>103,08</td><td rowspan=1 colspan=1>SH12542HF</td></tr></table>
"""

json_output = build_structured_table_json(
    table_img_path="",
    table_caption="This is my table caption",
    table_footnote=["This is my table footnote"],
    enhanced_caption="This is my enhanced table caption",
    table_body_html=table_html
)

# Print on terminal
# print(json_output)

# --- Write to file ---
with open("C:/Users/grabe_stud/llm_cad_projekt/RAG/llm_3d_cad/RAG-Anything/ollama-based/playground/pg_utils/table_2_json_output.txt", "w", encoding="utf-8") as f:
    f.write(json_output)
