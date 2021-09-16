# SVM-Modeling-Human-Cell-Records-Classification
SVM (Support Vector Machines) is using to build and train a model using human cell records, and classify cells to whether the samples are benign or malignant.

![](https://github.com/MinaMehrata/SVM-Modeling-Human-Cell-Records-Classification/blob/master/image/pic11.png)
![](https://github.com/MinaMehrata/SVM-Modeling-Human-Cell-Records-Classification/blob/master/Ipython.ipynb)
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>shop_id</th>
      <th>user_id</th>
      <th>order_amount</th>
      <th>total_items</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2500.500000</td>
      <td>50.078800</td>
      <td>849.092400</td>
      <td>3145.128000</td>
      <td>8.78720</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1443.520003</td>
      <td>29.006118</td>
      <td>87.798982</td>
      <td>41282.539349</td>
      <td>116.32032</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>607.000000</td>
      <td>90.000000</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1250.750000</td>
      <td>24.000000</td>
      <td>775.000000</td>
      <td>163.000000</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2500.500000</td>
      <td>50.000000</td>
      <td>849.000000</td>
      <td>284.000000</td>
      <td>2.00000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3750.250000</td>
      <td>75.000000</td>
      <td>925.000000</td>
      <td>390.000000</td>
      <td>3.00000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5000.000000</td>
      <td>100.000000</td>
      <td>999.000000</td>
      <td>704000.000000</td>
      <td>2000.00000</td>
    </tr>
  </tbody>
</table>








<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-MML-AM_CHTML-full,Safe"> </script>
    <!-- MathJax configuration -->
    <script type="text/x-mathjax-config">
    init_mathjax = function() {
        if (window.MathJax) {
        // MathJax loaded
            MathJax.Hub.Config({
                TeX: {
                    equationNumbers: {
                    autoNumber: "AMS",
                    useLabelIds: true
                    }
                },
                tex2jax: {
                    inlineMath: [ ['$','$'], ["\\(","\\)"] ],
                    displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
                    processEscapes: true,
                    processEnvironments: true
                },
                displayAlign: 'center',
                CommonHTML: {
                    linebreaks: { 
                    automatic: true 
                    }
                },
                "HTML-CSS": {
                    linebreaks: { 
                    automatic: true 
                    }
                }
            });
        
            MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
        }
    }
    init_mathjax();
    </script>
    <!-- End of mathjax configuration --></head>
<body class="jp-Notebook" data-jp-theme-light="true" data-jp-theme-name="JupyterLab Light">

<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p><strong>Question 1</strong>: Given some sample data, write a program to answer the following: click here to access the required data set</p>
<p>On Shopify, we have exactly 100 sneaker shops, and each of these shops sells only one model of shoe. We want to do some analysis of the average order value (AOV). When we look at orders data over a 30 day window, we naively calculate an AOV of $3145.13. Given that we know these shops are selling sneakers, a relatively affordable item, something seems wrong with our analysis.</p>
<p>Think about what could be going wrong with our calculation. Think about a better way to evaluate this data. 
What metric would you report for this dataset?
What is its value?</p>

</div>
</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h2 id="1-Importing-Data-and-Create-a-Data-Frame">1-Importing Data and Create a Data Frame<a class="anchor-link" href="#1-Importing-Data-and-Create-a-Data-Frame">&#182;</a></h2>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[1]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>
<span class="kn">import</span> <span class="nn">pandas</span> <span class="k">as</span> <span class="nn">pd</span> 
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[2]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">=</span><span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s1">&#39;data3.csv&#39;</span><span class="p">)</span>
</pre></div>

     </div>
</div>
</div>
</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[3]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">head</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[3]:</div>



<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>shop_id</th>
      <th>user_id</th>
      <th>order_amount</th>
      <th>total_items</th>
      <th>payment_method</th>
      <th>created_at</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>53</td>
      <td>746</td>
      <td>224</td>
      <td>2</td>
      <td>cash</td>
      <td>3/13/2017 12:36</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>92</td>
      <td>925</td>
      <td>90</td>
      <td>1</td>
      <td>cash</td>
      <td>3/3/2017 17:38</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>44</td>
      <td>861</td>
      <td>144</td>
      <td>1</td>
      <td>cash</td>
      <td>3/14/2017 4:23</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>18</td>
      <td>935</td>
      <td>156</td>
      <td>1</td>
      <td>credit_card</td>
      <td>3/26/2017 12:43</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>18</td>
      <td>883</td>
      <td>156</td>
      <td>1</td>
      <td>credit_card</td>
      <td>3/1/2017 4:35</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h2 id="2--Analysis-of-Data">2- Analysis of Data<a class="anchor-link" href="#2--Analysis-of-Data">&#182;</a></h2>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[4]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">info</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>


<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain">
<pre>&lt;class &#39;pandas.core.frame.DataFrame&#39;&gt;
RangeIndex: 5000 entries, 0 to 4999
Data columns (total 7 columns):
 #   Column          Non-Null Count  Dtype 
---  ------          --------------  ----- 
 0   order_id        5000 non-null   int64 
 1   shop_id         5000 non-null   int64 
 2   user_id         5000 non-null   int64 
 3   order_amount    5000 non-null   int64 
 4   total_items     5000 non-null   int64 
 5   payment_method  5000 non-null   object
 6   created_at      5000 non-null   object
dtypes: int64(5), object(2)
memory usage: 273.6+ KB
</pre>
</div>
</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[5]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">df</span><span class="o">.</span><span class="n">describe</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[5]:</div>



<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/html">
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order_id</th>
      <th>shop_id</th>
      <th>user_id</th>
      <th>order_amount</th>
      <th>total_items</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.000000</td>
      <td>5000.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2500.500000</td>
      <td>50.078800</td>
      <td>849.092400</td>
      <td>3145.128000</td>
      <td>8.78720</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1443.520003</td>
      <td>29.006118</td>
      <td>87.798982</td>
      <td>41282.539349</td>
      <td>116.32032</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>607.000000</td>
      <td>90.000000</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1250.750000</td>
      <td>24.000000</td>
      <td>775.000000</td>
      <td>163.000000</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2500.500000</td>
      <td>50.000000</td>
      <td>849.000000</td>
      <td>284.000000</td>
      <td>2.00000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3750.250000</td>
      <td>75.000000</td>
      <td>925.000000</td>
      <td>390.000000</td>
      <td>3.00000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5000.000000</td>
      <td>100.000000</td>
      <td>999.000000</td>
      <td>704000.000000</td>
      <td>2000.00000</td>
    </tr>
  </tbody>
</table>
</div>
</div>

</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[6]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="n">ax</span> <span class="o">=</span> <span class="n">df</span><span class="p">[</span><span class="s1">&#39;order_amount&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">plot</span><span class="o">.</span><span class="n">hist</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>

</div>

</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[32]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#Percentage of orders more than 1000</span>
<span class="n">np</span><span class="o">.</span><span class="n">sum</span><span class="p">(</span><span class="n">df</span><span class="p">[</span><span class="s1">&#39;order_amount&#39;</span><span class="p">]</span> <span class="o">&gt;</span> <span class="mi">3500</span><span class="p">)</span><span class="o">/</span><span class="mi">5000</span><span class="o">*</span><span class="mi">100</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[32]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>1.26</pre>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>Regards to  the hist plot and the statistical table,the order amount data is skewed(has outliers).</p>
<ol>
<li>" Order amount" standard deviation is almost 13 times of the mean value. </li>
<li>75% of the orders have values smaller than 390.</li>
<li>less than 2% of orders have a value more than 3500.
So, the mean (3145.128) can be misleading because the most common values in the distribution are not be near the mean. </li>
</ol>

</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[29]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="kn">from</span> <span class="nn">matplotlib</span> <span class="kn">import</span> <span class="n">pyplot</span> <span class="k">as</span> <span class="n">plt</span>
<span class="kn">import</span> <span class="nn">seaborn</span> <span class="k">as</span> <span class="nn">sns</span>

<span class="n">f</span><span class="p">,</span> <span class="p">(</span><span class="n">ax_box</span><span class="p">,</span> <span class="n">ax_hist</span><span class="p">)</span> <span class="o">=</span> <span class="n">plt</span><span class="o">.</span><span class="n">subplots</span><span class="p">(</span><span class="mi">2</span><span class="p">,</span> <span class="n">sharex</span><span class="o">=</span><span class="kc">True</span><span class="p">,</span> <span class="n">gridspec_kw</span><span class="o">=</span> <span class="p">{</span><span class="s2">&quot;height_ratios&quot;</span><span class="p">:</span> <span class="p">(</span><span class="mf">0.2</span><span class="p">,</span> <span class="mi">1</span><span class="p">)})</span>
<span class="n">ax_hist</span><span class="o">.</span><span class="n">set_xlim</span><span class="p">([</span><span class="mi">0</span><span class="p">,</span> <span class="mi">3500</span><span class="p">])</span>
<span class="n">mean</span><span class="o">=</span><span class="n">df</span><span class="p">[</span><span class="s1">&#39;order_amount&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">mean</span><span class="p">()</span>
<span class="n">median</span><span class="o">=</span><span class="n">df</span><span class="p">[</span><span class="s1">&#39;order_amount&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">median</span><span class="p">()</span>
<span class="n">mode</span><span class="o">=</span><span class="n">df</span><span class="p">[</span><span class="s1">&#39;order_amount&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">mode</span><span class="p">()</span><span class="o">.</span><span class="n">values</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>

<span class="n">sns</span><span class="o">.</span><span class="n">boxplot</span><span class="p">(</span><span class="n">data</span><span class="o">=</span><span class="n">df</span><span class="p">,</span> <span class="n">x</span><span class="o">=</span><span class="n">df</span><span class="p">[</span><span class="s1">&#39;order_amount&#39;</span><span class="p">],</span> <span class="n">ax</span><span class="o">=</span><span class="n">ax_box</span><span class="p">)</span>
<span class="n">ax_box</span><span class="o">.</span><span class="n">axvline</span><span class="p">(</span><span class="n">mean</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="s1">&#39;r&#39;</span><span class="p">,</span> <span class="n">linestyle</span><span class="o">=</span><span class="s1">&#39;--&#39;</span><span class="p">)</span>
<span class="n">ax_box</span><span class="o">.</span><span class="n">axvline</span><span class="p">(</span><span class="n">median</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="s1">&#39;g&#39;</span><span class="p">,</span> <span class="n">linestyle</span><span class="o">=</span><span class="s1">&#39;-&#39;</span><span class="p">)</span>
<span class="n">ax_box</span><span class="o">.</span><span class="n">axvline</span><span class="p">(</span><span class="n">mode</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="s1">&#39;b&#39;</span><span class="p">,</span> <span class="n">linestyle</span><span class="o">=</span><span class="s1">&#39;-&#39;</span><span class="p">)</span>

<span class="n">sns</span><span class="o">.</span><span class="n">histplot</span><span class="p">(</span><span class="n">data</span><span class="o">=</span><span class="n">df</span><span class="p">,</span> <span class="n">x</span><span class="o">=</span><span class="n">df</span><span class="p">[</span><span class="s1">&#39;order_amount&#39;</span><span class="p">],</span> <span class="n">ax</span><span class="o">=</span><span class="n">ax_hist</span><span class="p">,</span> <span class="n">kde</span><span class="o">=</span><span class="kc">True</span><span class="p">)</span>
<span class="n">ax_hist</span><span class="o">.</span><span class="n">axvline</span><span class="p">(</span><span class="n">mean</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="s1">&#39;r&#39;</span><span class="p">,</span> <span class="n">linestyle</span><span class="o">=</span><span class="s1">&#39;--&#39;</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="s2">&quot;Mean&quot;</span><span class="p">)</span>
<span class="n">ax_hist</span><span class="o">.</span><span class="n">axvline</span><span class="p">(</span><span class="n">median</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="s1">&#39;g&#39;</span><span class="p">,</span> <span class="n">linestyle</span><span class="o">=</span><span class="s1">&#39;-&#39;</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="s2">&quot;Median&quot;</span><span class="p">)</span>
<span class="n">ax_hist</span><span class="o">.</span><span class="n">axvline</span><span class="p">(</span><span class="n">mode</span><span class="p">,</span> <span class="n">color</span><span class="o">=</span><span class="s1">&#39;b&#39;</span><span class="p">,</span> <span class="n">linestyle</span><span class="o">=</span><span class="s1">&#39;-&#39;</span><span class="p">,</span> <span class="n">label</span><span class="o">=</span><span class="s2">&quot;Mode&quot;</span><span class="p">)</span>

<span class="n">ax_hist</span><span class="o">.</span><span class="n">legend</span><span class="p">()</span>

<span class="n">ax_box</span><span class="o">.</span><span class="n">set</span><span class="p">(</span><span class="n">xlabel</span><span class="o">=</span><span class="s1">&#39;&#39;</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt"></div>




<div class="jp-RenderedImage jp-OutputArea-output ">
<img src="data:image/png;base64,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
"
>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>The above picture only shows data in the range of 0 to 3500(around 98% of the data).The mean is realy far from the majority of data. The Median is a better metric to describe the order amount status.</p>

</div>
</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<h2 id="3--Better-Metric">3- Better Metric<a class="anchor-link" href="#3--Better-Metric">&#182;</a></h2>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[8]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#Calculate Median</span>
<span class="n">df</span><span class="p">[</span><span class="s1">&#39;order_amount&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">median</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[8]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>284.0</pre>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>Mode is also in a range of around 200.</p>

</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell   ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[31]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span><span class="c1">#Calculate Mode</span>
<span class="n">df</span><span class="p">[</span><span class="s1">&#39;order_amount&#39;</span><span class="p">]</span><span class="o">.</span><span class="n">mode</span><span class="p">()</span>
</pre></div>

     </div>
</div>
</div>
</div>

<div class="jp-Cell-outputWrapper">


<div class="jp-OutputArea jp-Cell-outputArea">

<div class="jp-OutputArea-child">

    
    <div class="jp-OutputPrompt jp-OutputArea-prompt">Out[31]:</div>




<div class="jp-RenderedText jp-OutputArea-output jp-OutputArea-executeResult" data-mime-type="text/plain">
<pre>0    153
dtype: int64</pre>
</div>

</div>

</div>

</div>

</div>
<div class="jp-Cell-inputWrapper"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput " data-mime-type="text/markdown">
<p>Summary: Only one statistical metric like “ MEAN” is not always enough to give a great overall view of data .For data with outliers   "MEDIAN” is more reasonable metric .</p>

</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs  ">
<div class="jp-Cell-inputWrapper">
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In&nbsp;[&nbsp;]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
     <div class="CodeMirror cm-s-jupyter">
<div class=" highlight hl-ipython3"><pre><span></span> 
</pre></div>

     </div>
</div>
</div>
</div>

</div>
</body>







</html>
