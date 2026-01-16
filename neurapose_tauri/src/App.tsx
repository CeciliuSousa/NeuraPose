import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Sidebar } from './components/layout/Sidebar';

// Pages
import HomePage from './pages/HomePage';
import ProcessamentoPage from './pages/ProcessamentoPage';
import ReidPage from './pages/ReidPage';
import AnotacaoPage from './pages/AnotacaoPage';
// import DatasetsPage from './pages/DatasetsPage';
import SplitPage from './pages/SplitPage';
import ConverterPage from './pages/ConverterPage';
import TreinoPage from './pages/TreinoPage';
import TestesPage from './pages/TestesPage';
import RelatoriosPage from './pages/RelatoriosPage';
import ConfiguracaoPage from './pages/ConfiguracaoPage';

function App() {
  return (
    <BrowserRouter>
      <div className="flex h-screen bg-background text-foreground">
        {/* Sidebar */}
        <Sidebar />

        {/* Main Content Area */}
        <main className="flex-1 ml-64 overflow-y-auto p-8">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/processamento" element={<ProcessamentoPage />} />
            <Route path="/reid" element={<ReidPage />} />
            <Route path="/anotacao" element={<AnotacaoPage />} />
            {/* <Route path="/datasets" element={<DatasetsPage />} /> */}
            <Route path="/split" element={<SplitPage />} />
            <Route path="/converter" element={<ConverterPage />} />
            <Route path="/treino" element={<TreinoPage />} />
            <Route path="/testes" element={<TestesPage />} />
            <Route path="/relatorios" element={<RelatoriosPage />} />
            <Route path="/configuracao" element={<ConfiguracaoPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
