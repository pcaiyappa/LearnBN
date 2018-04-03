
#create pdf reports of the cpts
# def write_cpts_as_pdf(cpts):
#     env = Environment(loader=FileSystemLoader('.'))
#     template = env.get_template("template.html")#use predefined template
#     template_vars = {"title": "Bayesian Networks Project","myvar":
#     pd.DataFrame.from_dict(cpts[list(cpts.keys())[1]]).to_html()}#create dict to pass as variable
#     html_out = template.render(template_vars)#pass dictionary to html renderer
#     HTML(string=html_out).write_pdf(f'{filename}_cpts.pdf')#use weasyprint to write pdf
#     print("Created pdf reports of the bayes net.")
#     return
