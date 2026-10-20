# Q&A agent: example traces

Picked from the evaluation run: the first answers that passed and were right
(or text answers that passed the checks), then the first that were wrong or sent
to review. Text answers have no answer key.

## Translate Bio, Inc. (2018): How many employees does the company have?

Status: **done** (checks passed)

> As of December 31, 2018, the company had 81 full-time employees, including a total of 25 employees with M.D., Pharm.D. or Ph.D. degrees. Of these full-time employees, 55 employees are engaged in research and development.

Citations: ['item1-132']  
Path: agent > tools > agent > verify > done

Trace:

1. `search_filing({"query": "number of employees"})` -> 5 chunks: ['item1a-069', 'item1-132', 'item1a-065', 'item15-076', 'item1a-094']

## Translate Bio, Inc. (2018): What was total revenue in fiscal 2018?

Status: **done** (checks passed)  
XBRL truth: 1,420,000.00 (in the text: False; correct within 5: True)

> Total revenue in fiscal 2018 was $1.4 million, which was derived from collaboration revenue under the Sanofi Agreement. The company had not generated any revenue from product sales as of December 31, 2018.

Citations: ['item7-017']  
Path: agent > tools > agent > tools > agent > tools > agent > tools > agent > tools > agent > tools > agent > verify > done

Trace:

1. `get_financials({})` -> status done, values {'net_income': -97400000.0, 'cash': 113200000.0}
2. `search_filing({"query": "total revenue fiscal 2018"})` -> 5 chunks: ['item15-075', 'item1a-137', 'item15-074', 'item15-054', 'item7-041']
3. `search_filing({"query": "revenue 2018 2017 net revenues"})` -> 5 chunks: ['item7-026', 'item7-017', 'item15-054', 'item15-053', 'item15-084']
4. `search_filing({"query": "collaboration revenue 1.4 million 2018"})` -> 5 chunks: ['item7-017', 'item7-008', 'item15-024', 'item7-022', 'item1-050']
5. `search_filing({"query": "total revenues year ended December 31 2018"})` -> 5 chunks: ['item7-019', 'item7-018', 'item15-077', 'item15-081', 'item7-020']
6. `search_filing({"query": "consolidated statements of operations revenues 2018"})` -> 5 chunks: ['item15-003', 'item7-041', 'item15-000', 'item7-045', 'item15-005']

## BioRestorative Therapies, Inc. (2016): What was total revenue in fiscal 2016?

Status: **done** (checks passed)  
XBRL truth: 36,355.00 (in the text: True; correct within 5: True)

> Total revenue in fiscal 2016 was $36,355, consisting of $36,000 from royalty revenue in connection with the sublicense agreement and $355 from sales of Stem Pearls skincare products.

Citations: ['item7-005']  
Path: agent > tools > agent > tools > agent > tools > agent > tools > agent > verify > done

Trace:

1. `get_financials({})` -> status done, values {'net_income': -8636292.0}
2. `search_filing({"query": "total revenue fiscal 2016"})` -> 5 chunks: ['item7-019', 'item15-028', 'item15-029', 'item7-020', 'item15-030']
3. `search_filing({"query": "revenue 2016"})` -> 5 chunks: ['item7-005', 'item15-026', 'item15-018', 'item15-021', 'item7-016']
4. `calculate({"expression": "36000 + 355"})` -> 36000 + 355 = 36355

## Translate Bio, Inc. (2018): Who are the company's largest customers, or how concentrated is its customer base?

Status: **done** (checks passed)

> The company's customer base is highly concentrated. Sanofi is the company's only customer. In fiscal year 2018, the company recognized collaboration revenue of $1.4 million from the Sanofi Agreement, which was the company's only source of revenue. The company had no revenue from product sales. The Sanofi Agreement, entered into in June 2018 and effective in July 2018, is a collaboration and license agreement to develop mRNA vaccines and mRNA vaccine platform development for up to five infectious disease pathogens. At December 31, 2018, the company had short-term and long-term deferred revenue of $2.6 million and $41.8 million, respectively, related to the Sanofi Agreement.

Citations: ['item7-017', 'item7-008', 'item15-024']  
Path: agent > tools > agent > tools > agent > tools > agent > tools > agent > tools > agent > tools > agent > verify > revise > agent > verify > done

Trace:

1. `search_filing({"query": "largest customers customer concentration", "k": 5})` -> 5 chunks: ['item7-037', 'item15-017', 'item15-023', 'item7-042', 'item15-039']
2. `search_filing({"query": "customers accounted for percentage revenue major customers"})` -> 5 chunks: ['item7-037', 'item15-017', 'item1-100', 'item1a-095', 'item15-055']
3. `search_filing({"query": "Sanofi revenue customer concentration single customer"})` -> 5 chunks: ['item7-042', 'item15-023', 'item7-043', 'item15-019', 'item7-009']
4. `search_filing({"query": "Sanofi 2018 revenue amount total revenue percentage"})` -> 5 chunks: ['item7-009', 'item15-053', 'item7-008', 'item15-024', 'item7-033']
5. `get_financials({})` -> status done, values {'net_income': -97400000.0, 'cash': 113200000.0}
6. `search_filing({"query": "total revenue 2018 collaboration revenue Sanofi million"})` -> 5 chunks: ['item7-017', 'item7-008', 'item7-022', 'item15-053', 'item15-024']
7. `verify({})` -> numbers not in a cited chunk or a calculation: ['100']

## BioRestorative Therapies, Inc. (2016): By what percent did total revenue change in fiscal 2016 versus the prior year?

Status: **needs_human_review** (checks failed after revision: calculation '(36355 - 628915) / 628915 * 100' uses numbers not found in the filing: ['36355', '628915', '628915']; numbers not in a cited chunk or a calculation: ['36,355', '628,915'])  
XBRL truth: -94.22 (in the text: False; correct within 5: True)

> Total revenue decreased by approximately 94.22% in fiscal 2016 versus the prior year. In 2016, total revenue was $36,355 ($36,000 from royalty revenue plus $355 from Stem Pearls skincare product sales), compared to $628,915 in 2015 ($609,490 from research and development agreements, $19,000 from royalty revenue, and $425 from Stem Pearls skincare product sales). The decrease was primarily due to the completion of research and development agreements as of December 31, 2015.

Citations: ['item7-005']  
Path: agent > tools > agent > tools > agent > verify > revise > agent > tools > agent > verify > needs_human_review

Trace:

1. `get_financials({})` -> status done, values {'net_income': -8636292.0}
2. `search_filing({"query": "total revenue fiscal 2016 2015", "k": 5})` -> 5 chunks: ['item15-028', 'item7-005', 'item7-018', 'item7-019', 'item15-021']
3. `calculate({"expression": "(36355 - 628915) / 628915 * 100"})` -> (36355 - 628915) / 628915 * 100 = -94.2194
4. `verify({})` -> calculation '(36355 - 628915) / 628915 * 100' uses numbers not found in the filing: ['36355', '628915', '628915']; numbers not in a cited chunk or a calculation: ['94.22', '36,355', '628,915']
5. `calculate({"expression": "(36000 + 355 - (609490 + 19000 + 425)) / (609490 + 19000 + 425) * 100"})` -> (36000 + 355 - (609490 + 19000 + 425)) / (609490 + 19000 + 425) * 100 = -94.2194

## BioRestorative Therapies, Inc. (2016): Who are the company's largest customers, or how concentrated is its customer base?

Status: **needs_human_review** (checks failed after revision: numbers not in a cited chunk or a calculation: ['10'])

> Based on the 10-K filing for fiscal year 2016, the company's customer base is highly concentrated:

**Fiscal Year 2016:** One license comprised substantially all of the company's revenue. Specifically, the company recognized $36,000 from royalty revenue in connection with its sublicense agreement (with SCTC - a stem cell treatment company) and $355 from sales of Stem Pearls skincare products.

**Fiscal Year 2015:** Two pharmaceutical clients comprised substantially all of the company's revenue. The company recognized $609,490 of revenues through research and development agreements with these clients, $19,000 from royalty revenue in connection with the sublicense agreement, and $425 from sales of Stem Pearls skincare products.

The company does not identify the specific names of the pharmaceutical clients from 2015, but the filing clearly indicates extreme customer concentration, with either one or two customers representing substantially all revenue in each year.

Citations: ['item15-018', 'item7-005', 'item15-020', 'item15-021']  
Path: agent > tools > agent > tools > agent > tools > agent > tools > agent > tools > agent > verify > revise > agent > verify > needs_human_review

Trace:

1. `search_filing({"query": "largest customers customer concentration", "k": 5})` -> 5 chunks: ['item1-008', 'item15-026', 'item7-019', 'item15-020', 'item7-016']
2. `search_filing({"query": "customer accounted for percentage revenue"})` -> 5 chunks: ['item7-090', 'item15-020', 'item7-019', 'item15-026', 'item7-075']
3. `search_filing({"query": "major customer single customer revenue concentration"})` -> 5 chunks: ['item15-020', 'item7-059', 'item1-008', 'item7-019', 'item7-081']
4. `search_filing({"query": "SCTC stem cell treatment company revenue royalty"})` -> 5 chunks: ['item15-020', 'item15-031', 'item15-018', 'item7-001', 'item15-021']
5. `search_filing({"query": "pharmaceutical clients revenue 2016 2015"})` -> 5 chunks: ['item15-018', 'item7-005', 'item15-021', 'item15-020', 'item15-033']
6. `verify({})` -> numbers not in a cited chunk or a calculation: ['10', '36,355', '628,915']

## BioRestorative Therapies, Inc. (2016): What are the main risk factors the company describes?

Status: **needs_human_review** (checks failed after revision: numbers not in a cited chunk or a calculation: ['10'])

> The main risk factors described by the company in its 2016 10-K filing include:

**Risks Related to Business Generally:**
- Limited operating history with substantial losses since inception and expectation of continued losses
- Working capital deficiency of $5,783,184 and stockholders' deficiency of $5,000,282 as of December 31, 2016
- Substantial doubt about ability to continue as a going concern
- High-risk business strategy focused on cellular-based products with commercial, legal, operational, scientific, and financial risks
- Dependence on attracting and retaining qualified management and scientific personnel
- Economic downturn impacts on funding availability

**Risks Related to Cell Therapy Product Development:**
- Challenges in enrolling sufficient patients in clinical trials
- Delays in completing clinical trials; management lacks significant experience in completing clinical trials and bringing drugs through commercialization
- Uncertainty due to inherent variability in autologous cell therapy manufacturing
- Lack of manufacturing capabilities at commercial scale and no alternate manufacturing supply
- Potential disruptions to access to media and reagents
- Clinical trials may fail to demonstrate safety and efficacy
- Anticipated delay of at least five years before commercializing BRTX-100

**Risks Related to Regulatory Approval:**
- FDA premarket clearance and approval process is lengthy, expensive, and uncertain
- Biosimilar competition and uncertainty regarding 12-year exclusivity period for biological products
- Potential product liability claims and inadequate insurance coverage
- Regulatory compliance requirements for manufacturing changes

**Risks Related to Market Acceptance:**
- Unknown commercial potential and profitability
- Market may not understand or accept cell therapy products
- Ethical and political controversies associated with stem cell products
- Reimbursement uncertainty from private and governmental insurers
- Healthcare reform impacts on reimbursement rates

**Risks Related to Intellectual Property:**
- Inability to protect proprietary rights and patents
- Potential disputes over intellectual property
- Uncertainty regarding patent protection in foreign countries
- Changes to U.S. patent law

**Risks Related to Competition:**
- Competition from companies with greater financial and technological resources
- Competitors may develop alternative products that are more effective or economical
- Competitors may obtain patent protection or FDA approval earlier

**Risks Related to Common Stock:**
- Stock price volatility and potential for significant fluctuations
- Penny stock classification with restricted liquidity
- Potential dilution from future issuances of common stock
- Thin and erratic trading on OTCQB market
- Potential future issuances of securities convertible into common stock

Citations: ['item7-021', 'item7-025', 'item7-027', 'item7-032', 'item7-033', 'item7-034', 'item7-036', 'item7-037', 'item7-038', 'item7-039', 'item7-041', 'item7-042', 'item7-049', 'item7-050', 'item7-054', 'item7-058', 'item7-073', 'item7-077', 'item7-080', 'item7-081', 'item7-084', 'item7-085', 'item7-087']  
Path: agent > tools > agent > tools > agent > tools > agent > tools > agent > tools > agent > tools > agent > verify > revise > agent > verify > needs_human_review

Trace:

1. `search_filing({"query": "risk factors", "k": 8})` -> 8 chunks: ['item1a-000', 'item7-021', 'item7-025', 'item7-049', 'item7-050', 'item1-029', 'item7-039', 'item1-038']
2. `search_filing({"query": "risk factors regulatory intellectual property competition", "k": 8})` -> 8 chunks: ['item1a-000', 'item7-045', 'item7-058', 'item7-021', 'item15-030', 'item7-020', 'item7-084', 'item7-049']
3. `search_filing({"k": 8, "query": "risk factors funding capital requirements clinical trials FDA approval"})` -> 8 chunks: ['item1a-000', 'item7-033', 'item7-034', 'item7-073', 'item7-021', 'item1-038', 'item7-032', 'item7-036']
4. `search_filing({"k": 8, "query": "risk factors key personnel management employees manufacturing supply chain"})` -> 8 chunks: ['item1a-000', 'item7-037', 'item7-021', 'item1-013', 'item7-038', 'item7-033', 'item7-027', 'item1-048']
5. `search_filing({"k": 8, "query": "risk factors market acceptance reimbursement competition ethical concerns"})` -> 8 chunks: ['item1a-000', 'item7-039', 'item7-041', 'item7-021', 'item7-054', 'item7-077', 'item7-080', 'item7-042']
6. `search_filing({"k": 8, "query": "risk factors dilution stock price volatility securities regulation"})` -> 8 chunks: ['item1a-000', 'item7-084', 'item7-021', 'item7-085', 'item15-022', 'item7-087', 'item15-060', 'item7-081']
7. `verify({})` -> numbers not in a cited chunk or a calculation: ['1505497', '10', '510']

## DXP ENTERPRISES INC (2019): What are the main risk factors the company describes?

Status: **needs_human_review** (checks failed after revision: numbers not in a cited chunk or a calculation: ['10'])

> Based on DXP's 2019 10-K filing, the main risk factors the company describes include:

1. **Energy Industry Dependence**: Decreased capital expenditures in the energy industry can adversely impact customer demand for products and services, as a significant portion of revenue depends on the level of capital and operating expenditures in the oil and gas sector.

2. **Economic and Industry Cyclicality**: Demand for products is subject to economic trends affecting customers and industries, particularly the oil and gas industry which is volatile, and the petrochemical and construction industries which are cyclical and materially affected by changes in the economy.

3. **Competition and Consolidation**: The company faces competition from other distributors, and consolidation in the industry could heighten competitive impacts. The company does not traditionally enter into long-term contracts with suppliers or customers, which increases competitive pressure.

4. **Acquisition Strategy Risks**: The company's future results depend on successfully implementing its acquisition strategy. Risks include difficulty identifying and completing acquisitions, high valuations, competition among buyers, regulatory approvals, credit market instability, and challenges integrating acquired businesses.

5. **Debt and Refinancing Risks**: The company may not be able to refinance existing debt on favorable terms or comply with financial covenants in credit facilities, which could result in default.

6. **Key Personnel and Talent Retention**: The loss of executive officers or failure to attract and retain qualified management and technical personnel could materially adversely affect financial condition and results of operations.

7. **Key Supplier Dependence**: Termination or limitation of relationships with key suppliers could disrupt business and adversely affect results of operations.

8. **Product Liability and Warranty Claims**: Customers use some products in potentially harmful and high-risk applications that may cause personal injury, loss of life, or property damage, creating significant product liability and warranty claim risks.

9. **Cyber Security Risks**: The company collects and maintains confidential customer, supplier, and employee information that could be compromised through cyber attacks or internal breaches, exposing the company to claims and reputational damage.

10. **Information Systems Disruptions**: Interruptions in information systems could disrupt operations and cause increases in costs or decreases in revenues.

11. **Goodwill and Intangible Asset Impairment**: As of December 31, 2019, combined goodwill and intangible assets amounted to $246.6 million. If the company does not generate sufficient cash flows to recover these investments, they could be impaired and subject to write-off, directly impacting earnings.

12. **Foreign Operations and Currency Risks**: The company has meaningful operations in Canada and Dubai, exposing it to foreign currency fluctuations, foreign laws and regulations, including import/export requirements, the FCPA, tax laws, data privacy requirements, and labor laws.

13. **Stock Price Volatility**: The market price of common stock could be subject to wide fluctuations in response to risk factors and other factors beyond the company's control.

14. **Pandemic and Public Health Crises**: Disruptions from pandemics or public health crises could impact supply chains, decrease customer demand, lower oil prices, and negatively affect operations and financial conditions.

Citations: ['item1a-000', 'item1a-005', 'item1a-006', 'item1a-008', 'item1a-009', 'item1a-011', 'item1a-012', 'item1a-014', 'item1a-015', 'item1a-016', 'item1a-017', 'item1a-018']  
Path: agent > tools > agent > tools > agent > tools > agent > tools > agent > tools > agent > tools > agent > verify > revise > agent > verify > needs_human_review

Trace:

1. `search_filing({"query": "risk factors", "k": 8})` -> 8 chunks: ['item1a-000', 'item1a-009', 'item7-000', 'item1a-014', 'item8-012', 'item8-009', 'item1a-018', 'item8-002']
2. `search_filing({"k": 8, "query": "risk factors competition customers suppliers"})` -> 8 chunks: ['item1a-000', 'item1a-009', 'item1a-018', 'item7-000', 'item1a-011', 'item1-002', 'item1a-012', 'item8-007']
3. `search_filing({"k": 8, "query": "risk factors debt financing liquidity economic conditions"})` -> 8 chunks: ['item1a-000', 'item1a-014', 'item7-000', 'item1a-009', 'item7-018', 'item7-019', 'item7-016', 'item1a-008']
4. `search_filing({"k": 8, "query": "risk factors key personnel employees retention management"})` -> 8 chunks: ['item1a-000', 'item1a-012', 'item7-000', 'item1a-009', 'item1a-006', 'item1a-005', 'item8-012', 'item1a-018']
5. `search_filing({"k": 8, "query": "risk factors technology systems information security cyber"})` -> 8 chunks: ['item1a-000', 'item1a-017', 'item7-000', 'item1a-009', 'item1a-018', 'item1a-016', 'item1a-015', 'item1-004']
6. `search_filing({"k": 8, "query": "risk factors inventory obsolescence goodwill impairment"})` -> 8 chunks: ['item1a-000', 'item8-009', 'item1a-015', 'item8-008', 'item1a-009', 'item7-000', 'item8-010', 'item8-027']
7. `verify({})` -> numbers not in a cited chunk or a calculation: ['10']
