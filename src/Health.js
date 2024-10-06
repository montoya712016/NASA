import React from "react";
import { FaHome, FaChartBar, FaMap, FaQuestionCircle, FaCog, FaChevronRight } from "react-icons/fa";
import { useNavigate } from "react-router-dom";

function Health() {
    const navigate = useNavigate();

    const handleClickHome = () => {
      navigate('/'); // Redireciona para /home
    };
  
    const handleClickDash = () => {
      navigate('/dashboard'); // Redireciona para /home
    };
  
    const handleClickHealth = () => {
      navigate('/health'); // Redireciona para /home
    };

    const handleClickFaq = () => {
        navigate('/faq'); // Redireciona para /faq
      };
    
  return (
    <div className="h-screen flex" style={{ backgroundColor: '#171717E5', fontWeight: 500 }}>
      {/* Sidebar */}
      <aside className="min-w-80 h-screen flex flex-col" style={{ backgroundColor: '#232323', color: '#828282', paddingLeft: '1rem' }}>
        {/* Logo */}
        <div className="flex justify-center -mb-4 ">
          <img src="/images/logo.png" alt="Logo" className="h-40 w-auto" />
        </div>

        {/* Sidebar Items */}
        <ul className="space-y-8 ml-8">
          <li className="flex items-center justify-between w-full cursor-pointer" onClick={handleClickHome}>
            <div className="flex items-center space-x-4">
              <FaHome className="text-gray-400" size={24} />
              <span style={{ color: '#828282', fontSize: '18px' }}>Home</span>
            </div>
            <FaChevronRight className="text-gray-400" size={16} style={{ marginLeft: 'auto', marginRight: '25px' }} />
          </li>
          <li className="flex items-center justify-between w-full cursor-pointer" onClick={handleClickDash}>
            <div className="flex items-center space-x-4">
              <FaChartBar className="text-gray-400" size={24} />
              <span style={{ color: '#828282', fontSize: '18px' }}>Dashboard</span>
            </div>
            <FaChevronRight className="text-gray-400" size={16} style={{ marginLeft: 'auto', marginRight: '25px' }} />
          </li>
          <li className="flex items-center justify-between w-full cursor-pointer" onClick={handleClickHealth}>
            <div className="flex items-center space-x-4">
              <FaMap className="text-green-500" size={24} />
              <span style={{ color: '#1C6E14', fontSize: '18px' }}>Health Map</span>
            </div>
            <FaChevronRight className="text-gray-400" size={16} style={{ marginLeft: 'auto', marginRight: '25px' }} />
          </li>
          <li className="flex items-center justify-between w-full cursor-pointer" onClick={handleClickFaq}>
            <div className="flex items-center space-x-4">
              <FaQuestionCircle className="text-gray-400" size={24} />
              <span style={{ color: '#828282', fontSize: '18px' }}>Help</span>
            </div>
            <FaChevronRight className="text-gray-400" size={16} style={{ marginLeft: 'auto', marginRight: '25px' }} />
          </li>

        </ul>

      </aside>

      {/* Main content with full-screen image */}
      <main className="h-screen w-screen">
        <img 
          src="/images/backgraph.png" // Caminho correto da imagem
          alt="Field Map"
          className="w-full h-full object-cover"
        />
        <img 
          src="/images/details.png" // Caminho correto da imagem
          alt="Field Map"
          className="absolute w-64 bottom-4 right-4"
        />
        <img 
          src="/images/mini-graph.png" // Caminho correto da imagem
          alt="Field Map"
          className="absolute w-96 bottom-0 left-80"
        />
      </main>
    </div>
  );
}

export default Health;
