
PROJECT TITLE





Bo-Shen Chen
boshenchentnt@gmail.com
Tatung University
Yi-Wen Chu
william1006.chu@gmail.com
Iowa State University/ 
Tatung University
Barry Yu
barry.yu@mail.mcgill.ca
McGill University
Andy S. Yu
asyu@uwaterloo.ca
University of Waterloo

With
Apart Research

Abstract
Summarize your project in 150–250 words. A strong abstract lets a reviewer understand what you did and why it matters without reading anything else. Make sure to cover: the problem, your approach, key results, and the main takeaway. Polish it last: the abstract should reflect your final results, not your initial plan.





How to use this template: Replace the italicized guidance text under each section with your content. The section structure is strong guidance but not rigid. If your project requires a different organization, feel free to adapt. Delete all guidance text including this info box before submitting. Make sure the project title and author information above are up to date.
How your submission will be evaluated: Your project will be judged on the quality of this written report. To understand what we look for, see our Evaluation Rubric.
Recommended length: 4 pages excluding references and appendix. 
Rough guide: Intro & Related Work 1p, Methods and Results: 2.5p, Discussion 0.5p.

1. Introduction
What problem are you addressing and why does it matter? 
Connect it to your work: we want to know why your work is practically valuable. 

Provide enough background for readers to understand your work. 
If relevant, briefly describe the threat model or failure mode you're addressing; reference prior work that motivates why it is worth addressing, or explain it yourself. 

Aspire to clearly list your most important contributions that go beyond what exists today.
Our main contributions are:

[First contribution — what new thing did you create, discover, or demonstrate?]
[Second contribution]
[Third contribution, if applicable]

2. Related Work
What prior work is most similar, and how does your work differ? 
Cite the most relevant papers, tools, or projects. Explain what gap your work addresses.

Some questions which may help:
When and why would someone use your method over the existing state-of-the-art?
What information/insight does your method provide which we did not have before?

Representation probing. Probing is a class of techniques wherein a model is used to predict properties from vector representations [1]. A key subcategory is the decoding of LLM latent states as distributions over model vocabulary (LogitLens [2], TunedLens [3], PatchScopes [4], and J-lens [5]). By interpreting lens readouts as parts of a reasoning trajectory (https://arxiv.org/abs/2303.08112)[6], past research has applied the lens across layers to probe for the “thought process” of a language model (https://arxiv.org/pdf/2307.09476)(https://arxiv.org/abs/2402.16837). 


J-lens is commonly used to provide information about an LLM’s layer-by-layer computations
 and is seen as a form of interpretation of a model’s “thought process” [6]. 


Emergence of preference systems. [llm pref] Recent work has demonstrated that LLMs possess partially consistent preference and value systems towards pairs of outcomes, and that this consistency increases with model scale (https://arxiv.org/abs/2502.08640). However, research has also shown that these preferences are not necessarily consistent with model behavior in more practical contexts (https://arxiv.org/abs/2606.11016) (https://arxiv.org/abs/2606.22974). In our work, [something]

3. Methods
Describe your approach clearly enough that someone could replicate it. Include key design choices and justify them where relevant (hint: the more you can back up your design choices by referencing prior work, the better).

What models/datasets/tools did you use and why? What were key parameters or design decisions? What did you try that didn't work? Could someone reproduce your work from this description?

4. Results
Present your main findings with appropriate evidence. Use figures and tables where appropriate (we strongly encourage at least one figure, see tips below). Distinguish between observations and interpretations. 

Argue why your claims are robust. E.g., if your approach “performs better” than alternatives:
Do you have enough data? Is the difference statistically significant?
Is it robust? Or do small changes to your setup cause substantial changes to your results?

Tips for figures and tables:

Number all figures (Figure 1, Figure 2...) and tables (Table 1, Table 2...)
Include descriptive captions that can be understood without the main text
Place figures/tables near where they're first referenced
Ensure text in figures is legible!

5. Discussion and Limitations
Discuss the broader implications for AI safety. 
What do your results mean? What trends do you notice and what might they indicate?  

Limitations
What are the limitations of your work? What threat models or failure modes did you not address? Be honest about constraints — methodological limitations, scope limitations, or aspects you couldn't fully address in the hackathon timeframe. Explicitly note the assumptions you made, whether implicitly or explicitly, and how the interpretation of your results would change if a given assumption did not hold.

Future Work
What are the natural next steps? How could this work be extended?

6. Conclusion
Briefly summarize your main findings and their implications (1–2 paragraphs).

Code and Data
Include links if applicable. If your project doesn't involve code (e.g., policy analysis) or if there are info-hazard considerations, note that here.

Code repository: [Link to GitHub/GitLab if applicable]
Data/Datasets: [Link if applicable]
Other artifacts (optional): [Demo link, video walkthrough, Hugging Face Space, etc.]

Author Contributions (optional) 
[e.g., "A.B. led the project and designed experiments. C.D. implemented the code. All authors contributed to writing and reviewed the final manuscript."]

References
Use a consistent citation format. Include: Author(s), Year, Title, Venue/Publisher, and URL or DOI where available.
[1]	J. Hewitt and P. Liang, “Designing and Interpreting Probes with Control Tasks,” 2019, arXiv. doi: 10.48550/ARXIV.1909.03368. 
[2]	nostalgebraist, “interpreting GPT: the logit lens,” LessWrong. Accessed: Aug. 15, 2026. [Online]. Available: https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens 
[3]	N. Belrose et al., “Eliciting Latent Predictions from Transformers with the Tuned Lens,” Nov. 11, 2025, arXiv: arXiv:2303.08112. doi: 10.48550/arXiv.2303.08112. 
[4]	A. Ghandeharioun, A. Caciularu, A. Pearce, L. Dixon, and M. Geva, “Patchscopes: A Unifying Framework for Inspecting Hidden Representations of Language Models,” Jun. 06, 2024, arXiv: arXiv:2401.06102. doi: 10.48550/arXiv.2401.06102. 
[5]	W. Gurnee et al., “Verbalizable Representations Form a Global Workspace in Language Models,” Transform. Circuits Thread, 2026, [Online]. Available: https://transformer-circuits.pub/2026/workspace/index.html 
[6]	A. Y. Din, T. Karidi, L. Choshen, and M. Geva, “Jump to Conclusions: Short-Cutting Transformers With Linear Transformations,” Jun. 18, 2024, arXiv: arXiv:2303.09435. doi: 10.48550/arXiv.2303.09435. 



Appendix (optional)
Supplementary material such as additional figures, detailed methodology, prompts used, extended results, etc.
LLM Usage Statement
If you used LLM assistance in developing your project or writing this report, briefly note how. Ensure all claims and results have been verified.

NOTE: We strongly encourage that the final version of the submission is primarily written by your team. 

[e.g., "We used Claude to brainstorm approaches and help draft sections. All results and claims were independently verified."]

