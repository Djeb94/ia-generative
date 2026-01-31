import React, { useState } from 'react';
import './SkillsForm.css';

function SkillsForm() {
  const [formData, setFormData] = useState({
    // Likert
    dataCleaning: 3,
    dataVisualization: 3,
    machineLeaning: 3,
    statistics: 3,
    
    // Ouvertes
    background: '',
    achievements: '',
    
    // Choix
    primaryTool: '',
    
    // Cases
    skills: [],
    
    // Expérience
    experience: ''
  });

  const [submitted, setSubmitted] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleLikertChange = (field, value) => {
    setFormData({
      ...formData,
      [field]: value
    });
  };

  const handleTextChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // Préparer payload: texte libre + autres réponses
    const freeTextParts = [];
    if (formData.background) freeTextParts.push(formData.background);
    if (formData.achievements) freeTextParts.push(formData.achievements);
    if (formData.skills && formData.skills.length) freeTextParts.push(`Compétences: ${formData.skills.join(', ')}`);
    // Ajouter niveaux Likert comme texte
    freeTextParts.push(`Nettoyage:${formData.dataCleaning}, Visualisation:${formData.dataVisualization}, ML:${formData.machineLeaning}, Statistiques:${formData.statistics}`);

    const payload = {
      free_text: freeTextParts.join('\n'),
      other_answers: formData
    };
    console.log("Form data payload :", JSON.stringify(payload, null, 2));

    setLoading(true);
    setError(null);
    fetch('http://127.0.0.1:8000/analyse', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        setAnalysisResult(data);
        setSubmitted(true);
        setTimeout(() => setSubmitted(false), 3000);
      })
      .catch((err) => {
        console.error('Analyse error', err);
        setError('Erreur lors de l\'analyse.');
      })
      .finally(() => setLoading(false));
  };

  const likertLabels = ['Pas du tout', 'Peu', 'Modéré', 'Bon', 'Excellent'];
  

  const experienceOptions = [
    'Moins de 1 an',
    '1-2 ans',
    '2-5 ans',
    '5-10 ans',
    'Plus de 10 ans'
  ];

  return (
    <div className="skills-form-container">
      <div className="form-header">
        <h1>Évaluation des Compétences en Data Science</h1>
        <p>Aidez-nous à mieux comprendre votre profil et vos compétences</p>
      </div>

      <form onSubmit={handleSubmit} className="skills-form">
        {/* Likert */}
        <section className="form-section">
          <h2 className="section-title">📊 Compétences Techniques (Échelle de Likert)</h2>
          <p className="section-description">Évaluez votre niveau de maîtrise pour chaque domaine</p>

          <div className="likert-group">
            <div className="likert-item">
              <label className="likert-label">Nettoyage et préparation des données</label>
              <p className="likert-description">Exemple : Nettoyage de données avec Python, gestion des valeurs manquantes</p>
              <div className="likert-scale">
                {likertLabels.map((label, index) => (
                  <label key={index} className="likert-option">
                    <input
                      type="radio"
                      name="dataCleaning"
                      value={index}
                      checked={formData.dataCleaning === index}
                      onChange={(e) => handleLikertChange('dataCleaning', parseInt(e.target.value))}
                    />
                    <span className={`likert-value likert-${index}`}>{index}</span>
                    <span className="likert-text">{label}</span>
                  </label>
                ))}
              </div>
            </div>

            <div className="likert-item">
              <label className="likert-label">Visualisation des données</label>
              <p className="likert-description">Exemple : Création de dashboards interactifs avec Python, Tableau ou Power BI</p>
              <div className="likert-scale">
                {likertLabels.map((label, index) => (
                  <label key={index} className="likert-option">
                    <input
                      type="radio"
                      name="dataVisualization"
                      value={index}
                      checked={formData.dataVisualization === index}
                      onChange={(e) => handleLikertChange('dataVisualization', parseInt(e.target.value))}
                    />
                    <span className={`likert-value likert-${index}`}>{index}</span>
                    <span className="likert-text">{label}</span>
                  </label>
                ))}
              </div>
            </div>

            <div className="likert-item">
              <label className="likert-label">Machine Learning & Modélisation</label>
              <p className="likert-description">Exemple : Étude des modèles de régression, Deep Learning et algorithmes supervisés</p>
              <div className="likert-scale">
                {likertLabels.map((label, index) => (
                  <label key={index} className="likert-option">
                    <input
                      type="radio"
                      name="machineLeaning"
                      value={index}
                      checked={formData.machineLeaning === index}
                      onChange={(e) => handleLikertChange('machineLeaning', parseInt(e.target.value))}
                    />
                    <span className={`likert-value likert-${index}`}>{index}</span>
                    <span className="likert-text">{label}</span>
                  </label>
                ))}
              </div>
            </div>

            <div className="likert-item">
              <label className="likert-label">Statistiques & Analyse</label>
              <p className="likert-description">Exemple : Tests statistiques, probabilités, inférence statistique</p>
              <div className="likert-scale">
                {likertLabels.map((label, index) => (
                  <label key={index} className="likert-option">
                    <input
                      type="radio"
                      name="statistics"
                      value={index}
                      checked={formData.statistics === index}
                      onChange={(e) => handleLikertChange('statistics', parseInt(e.target.value))}
                    />
                    <span className={`likert-value likert-${index}`}>{index}</span>
                    <span className="likert-text">{label}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>
        </section>

                {/* Champ libre outil principal */}
        <section className="form-section">
          <h2 className="section-title">🔧 Outils principaux utilisés et dans quel but</h2>
          <p className="section-description">
            Décrivez librement les outils que vous utilisez pour l’analyse de données
          </p>

          <textarea
            name="primaryTool"
            value={formData.primaryTool}
            onChange={handleTextChange}
            placeholder="Ex : Python (Pandas, NumPy), Power BI, Excel, SQL, Tableau, TensorFlow..."
            className="textarea"
            rows={4}
          />
        </section>

        {/* Expérience */}
        <section className="form-section">
          <h2 className="section-title">📅 Expérience Professionnelle (Questions guidées)</h2>
          <p className="section-description">Combien de temps avez-vous travaillé dans le domaine de la data science ?</p>
          
          <div className="guided-options">
            {experienceOptions.map((option) => (
              <label key={option} className="guided-option">
                <input
                  type="radio"
                  name="experience"
                  value={option}
                  checked={formData.experience === option}
                  onChange={handleTextChange}
                />
                <span className="guided-label">{option}</span>
              </label>
            ))}
          </div>
        </section>

        {/* Ouvertes */}
        <section className="form-section">
          <h2 className="section-title">📝 Questions Ouvertes</h2>
          <p className="section-description">Partagez vos expériences et apprentissages en détail</p>

          <div className="textarea-group">
            <label className="textarea-label">
              Décrivez votre formation et votre parcours en data science
              <textarea
                name="background"
                value={formData.background}
                onChange={handleTextChange}
                placeholder="Exemple : J'ai suivi une formation en statistiques à l'université, puis j'ai travaillé 2 ans en tant que data analyst..."
                className="textarea"
              />
            </label>
          </div>

          <div className="textarea-group">
            <label className="textarea-label">
              Décrivez vos réalisations les plus importantes
              <textarea
                name="achievements"
                value={formData.achievements}
                onChange={handleTextChange}
                placeholder="Exemple : J'ai créé un modèle de prédiction qui a augmenté la précision de 15%, réalisé plusieurs dashboards interactifs..."
                className="textarea"
              />
            </label>
          </div>
        </section>

        {/* Envoyer */}
        <div className="form-footer">
          <button type="submit" className="submit-btn">
            {loading ? 'Envoi...' : '📤 Soumettre le formulaire'}
          </button>
          {submitted && (
            <div className="success-message">
              ✅ Formulaire soumis avec succès !
            </div>
          )}
        </div>
        {/* Résultats d'analyse */}
        {error && <div className="success-message" style={{background:'#fdecea',borderColor:'#e55353',color:'#6b1b1b'}}>⚠️ {error}</div>}
        {analysisResult && (
          <section className="form-section">
            <h3 className="section-title">Résultats d'analyse</h3>
            <div style={{marginTop:12}}>
              <strong>Compétences détectées :</strong>
              <ul>
                {analysisResult.matched_competences && analysisResult.matched_competences.length ? (
                  analysisResult.matched_competences.map(c => (
                    <li key={c.id}>{c.name} — score: {c.score.toFixed(3)}</li>
                  ))
                ) : <li>Aucune compétence détectée</li>}
              </ul>

              <strong>Scores par bloc :</strong>
              <ul>
                {analysisResult.block_scores && Object.keys(analysisResult.block_scores).length ? (
                  Object.entries(analysisResult.block_scores).map(([b, s]) => (
                    <li key={b}>{b}: {s}</li>
                  ))
                ) : <li>Aucun score</li>}
              </ul>

              <strong>Métiers recommandés :</strong>
              <ul>
                {analysisResult.recommended_jobs && analysisResult.recommended_jobs.length ? (
                  analysisResult.recommended_jobs.map((job, i) => (
                    <li key={i}>{job.job} — match: {job.match.toFixed(0)}%</li>
                  ))
                ) : <li>Aucune recommandation</li>}
              </ul>
            </div>
          </section>
        )}
      </form>
    </div>
  );
}

export default SkillsForm;
