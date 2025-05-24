import { Component, ElementRef, ViewChild, OnInit, OnDestroy, AfterViewInit, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import * as tf from '@tensorflow/tfjs';
import * as d3 from 'd3';

interface AIAgent {
  id: string;
  name: string;
  model: tf.Sequential;
  targetModel: tf.Sequential;
  memory: Experience[];
  epsilon: number;
  score: number;
  totalReward: number;
  gamesPlayed: number;
  color: string;
  // New optimization fields
  actionCache: Map<string, number>;
  lastState: number[] | null;
  lastAction: number;
}

interface Experience {
  state: number[];
  action: number;
  reward: number;
  nextState: number[];
  done: boolean;
}

interface GameState {
  ballX: number;
  ballY: number;
  ballDx: number;
  ballDy: number;
  paddle1Y: number;
  paddle2Y: number;
  player1Score: number;
  player2Score: number;
}

interface LearningConfig {
  learningRate: number;
  epsilon: number;
  epsilonDecay: number;
  epsilonMin: number;
  batchSize: number;
  memorySize: number;
  targetUpdateFreq: number;
  gameSpeed: number;
  episodes: number;
  // New optimization settings
  stepsPerFrame: number;
  trainFrequency: number;
  prioritizedReplay: boolean;
  doubleQLearning: boolean;
  maxGameLength: number;
}

interface TrainingLog {
  timestamp: string;
  gameNumber: number;
  agent1Score: number;
  agent2Score: number;
  agent1Reward: number;
  agent2Reward: number;
  agent1Epsilon: number;
  agent2Epsilon: number;
  message: string;
}

@Component({
  selector: 'app-ai-learning',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './ai-learning.component.html',
  styleUrls: ['./ai-learning.component.css']
})
export class AiLearningComponent implements OnInit, OnDestroy, AfterViewInit {
  @ViewChild('gameCanvas', { static: true }) canvasRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('networkVisualization', { static: true }) networkVisRef!: ElementRef<HTMLDivElement>;
  @Output() backToMenu = new EventEmitter<void>();

  private ctx!: CanvasRenderingContext2D;
  private animationId!: number;
  isTraining = false;
  gameNumber = 0;
  episode = 0;

  // Game configuration
  canvas = { width: 800, height: 400 };
  ball = { x: 400, y: 200, dx: 6, dy: 4, radius: 8 }; // Increased speed from 4,3 to 6,4
  paddle1 = { x: 20, y: 175, width: 10, height: 80 };
  paddle2 = { x: 770, y: 175, width: 10, height: 80 };
  
  // AI Agents
  agent1: AIAgent | null = null;
  agent2: AIAgent | null = null;

  // Training logs and tracking
  trainingLogs: TrainingLog[] = [];
  maxLogs = 100;

  // Configuration for episodes
  config: LearningConfig = {
    learningRate: 0.003, // Increased from 0.001
    epsilon: 0.9, // Reduced from 1.0 for faster exploitation
    epsilonDecay: 0.998, // Faster decay
    epsilonMin: 0.05, // Higher minimum for continued exploration
    batchSize: 64, // Increased from 32
    memorySize: 20000, // Increased capacity
    targetUpdateFreq: 50, // More frequent updates
    gameSpeed: 5, // Higher default speed
    episodes: 1000,
    // New optimization settings
    stepsPerFrame: 10, // Multiple game steps per frame
    trainFrequency: 4, // Train every 4 steps instead of every step
    prioritizedReplay: true,
    doubleQLearning: true,
    maxGameLength: 1000 // Prevent infinite games
  };

  // Training statistics
  stats = {
    agent1Wins: 0,
    agent2Wins: 0,
    totalGames: 0,
    avgRewardAgent1: 0,
    avgRewardAgent2: 0,
    agent1WinRate: 0,
    agent2WinRate: 0,
    agent1Rewards: [] as number[],
    agent2Rewards: [] as number[]
  };

  // Visualization
  private readonly lossHistory: number[] = [];
  private readonly rewardHistory: { agent1: number[], agent2: number[] } = { agent1: [], agent2: [] };
  private readonly winRateHistory: { agent1: number[], agent2: number[] } = { agent1: [], agent2: [] };

  private readonly networkSvg: any;
  private readonly networkConfig = {
    layers: [8, 128, 64, 32, 3],
    nodeRadius: 8,
    layerSpacing: 120,
    nodeSpacing: 25
  };

  // Performance optimization fields
  private frameCount = 0;
  private lastFrameTime = 0;
  private fps = 0;
  private gameStepCount = 0;
  private readonly renderSkipFrames = 5; // Render every 5th frame for better performance
  private readonly trainingBuffer: { agent: AIAgent, state: number[], action: number, reward: number, nextState: number[], done: boolean }[] = [];

  ngOnInit() {
    this.initializeCanvas();
  }

  ngAfterViewInit() {
    this.setupVisualization();
  }

  ngOnDestroy() {
    this.stopTraining();
    this.cleanup();
  }

  private initializeCanvas() {
    const canvas = this.canvasRef.nativeElement;
    canvas.width = this.canvas.width;
    canvas.height = this.canvas.height;
    this.ctx = canvas.getContext('2d')!;
  }

  private async setupVisualization() {
    // Initialize TensorFlow.js visualization
    await tf.ready();
    console.log('TensorFlow.js ready for AI training!');
    
    // Initialize D3.js neural network visualization
    setTimeout(() => {
      this.initializeNetworkVisualization();
    }, 100);
  }

  private createNeuralNetwork(): tf.Sequential {
    // Optimized smaller network for faster training
    const model = tf.sequential({
      layers: [
        tf.layers.dense({ 
          inputShape: [8], 
          units: 64, // Reduced from 128
          activation: 'relu',
          kernelInitializer: 'heNormal' // Better initialization
        }),
        tf.layers.batchNormalization(), // Add batch normalization
        tf.layers.dropout({ rate: 0.1 }), // Reduced dropout
        tf.layers.dense({ 
          units: 32, // Reduced from 64
          activation: 'relu',
          kernelInitializer: 'heNormal'
        }),
        tf.layers.batchNormalization(),
        tf.layers.dense({ 
          units: 3, 
          activation: 'linear',
          kernelInitializer: 'zeros' // Start with zero Q-values
        })
      ]
    });

    model.compile({
      optimizer: tf.train.adam(this.config.learningRate, 0.9, 0.999, 1e-7), // Optimized Adam
      loss: 'huberLoss', // More stable than MSE
      metrics: ['mse']
    });

    return model;
  }

  // High-performance training loop with batched steps
  private async trainingLoop() {
    if (!this.isTraining || !this.agent1 || !this.agent2) return;

    const startTime = performance.now();

    // Run multiple game steps per frame for higher FPS
    for (let i = 0; i < this.config.stepsPerFrame; i++) {
      await this.gameStep();
      this.gameStepCount++;

      // Early break if game ended
      if (this.gameNumber !== Math.floor(this.gameStepCount / 100)) {
        break;
      }
    }

    // Process training buffer in batches for efficiency
    if (this.trainingBuffer.length >= this.config.batchSize && this.gameStepCount % this.config.trainFrequency === 0) {
      await this.processBatchedTraining();
    }

    // Render less frequently for better performance
    this.frameCount++;
    if (this.frameCount % this.renderSkipFrames === 0) {
      this.render();
      this.updateFPS(startTime);
    }

    // Update visualization less frequently
    if (this.gameNumber % 20 === 0 && this.gameNumber > 0) {
      this.updateVisualization();
    }

    // Continue training
    this.animationId = requestAnimationFrame(() => this.trainingLoop());
  }

  // Enhanced start training with better logging
  async startTraining() {
    if (this.isTraining) return;

    this.isTraining = true;
    this.gameNumber = 0;
    this.episode = 0;
    this.gameStepCount = 0;
    this.frameCount = 0;

    this.addTrainingLog('🚀 High-performance training session started');
    this.addTrainingLog('🧠 Initializing optimized neural networks...');

    // Initialize AI agents with optimizations
    this.agent1 = {
      id: 'agent1',
      name: 'Red AI',
      model: this.createNeuralNetwork(),
      targetModel: this.createNeuralNetwork(),
      memory: [],
      epsilon: this.config.epsilon,
      score: 0,
      totalReward: 0,
      gamesPlayed: 0,
      color: '#ff4444',
      actionCache: new Map(),
      lastState: null,
      lastAction: 1
    };

    this.agent2 = {
      id: 'agent2',
      name: 'Blue AI',
      model: this.createNeuralNetwork(),
      targetModel: this.createNeuralNetwork(),
      memory: [],
      epsilon: this.config.epsilon,
      score: 0,
      totalReward: 0,
      gamesPlayed: 0,
      color: '#4444ff',
      actionCache: new Map(),
      lastState: null,
      lastAction: 1
    };

    // Copy initial weights to target networks
    this.agent1.targetModel.setWeights(this.agent1.model.getWeights());
    this.agent2.targetModel.setWeights(this.agent2.model.getWeights());

    this.addTrainingLog('✅ Optimized agents initialized successfully');
    this.addTrainingLog(`🎮 Running at ${this.config.stepsPerFrame}x speed multiplier`);

    this.resetGame();
    this.trainingLoop();
  }

  // Enhanced stop training with logging
  stopTraining() {
    this.isTraining = false;
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
    }
    
    if (this.gameNumber > 0) {
      this.addTrainingLog(`⏹️ Training stopped after ${this.gameNumber} games`);
      const agent1WinRate = this.stats.agent1WinRate.toFixed(1);
      const agent2WinRate = this.stats.agent2WinRate.toFixed(1);
      this.addTrainingLog(`📊 Final win rates - Red: ${agent1WinRate}%, Blue: ${agent2WinRate}%`);
    }
  }

  private async gameStep() {
    if (!this.agent1 || !this.agent2) return;

    // Get current state
    const state = this.getGameState();

    // Get actions from both agents with caching
    const action1 = await this.getActionOptimized(this.agent1, state);
    const action2 = await this.getActionOptimized(this.agent2, state);

    // Store previous state
    const prevState = [...state];

    // Execute actions
    this.executeAction(1, action1);
    this.executeAction(2, action2);

    // Update game physics multiple times for faster games
    for (let i = 0; i < this.config.gameSpeed; i++) {
      this.updateBall();
      const gameResult = this.checkGameEnd();
      
      if (gameResult.gameEnded) {
        // Get new state and calculate rewards
        const newState = this.getGameState();
        const rewards = this.calculateEnhancedRewards(gameResult, prevState, newState);

        // Add to training buffer instead of immediate processing
        this.trainingBuffer.push(
          { agent: this.agent1, state: prevState, action: action1, reward: rewards.agent1, nextState: newState, done: true },
          { agent: this.agent2, state: prevState, action: action2, reward: rewards.agent2, nextState: newState, done: true }
        );

        this.handleGameEnd(gameResult.winner);
        this.resetGame();
        this.gameNumber++;
        return;
      }
    }

    // Get new state and calculate rewards for ongoing game
    const newState = this.getGameState();
    const rewards = this.calculateEnhancedRewards({ gameEnded: false, winner: null }, prevState, newState);

    // Add to training buffer
    this.trainingBuffer.push(
      { agent: this.agent1, state: prevState, action: action1, reward: rewards.agent1, nextState: newState, done: false },
      { agent: this.agent2, state: prevState, action: action2, reward: rewards.agent2, nextState: newState, done: false }
    );

    this.episode++;
    if (this.episode % this.config.targetUpdateFreq === 0) {
      this.updateTargetNetworks();
    }
  }

  // Optimized action selection with caching
  private async getActionOptimized(agent: AIAgent, state: number[]): Promise<number> {
    // Create state key for caching
    const stateKey = state.map(x => Math.round(x * 100) / 100).join(',');
    
    // Check cache first
    if (agent.actionCache.has(stateKey) && Math.random() > 0.1) {
      return agent.actionCache.get(stateKey)!;
    }

    // Epsilon-greedy with optimizations
    if (Math.random() < agent.epsilon) {
      // Smart random action - prefer actions that move toward ball
      const ballY = (state[1] + 1) * this.canvas.height / 2;
      const paddleY = agent.id === 'agent1' ? 
        (state[4] + 1) * this.canvas.height / 2 : 
        (state[5] + 1) * this.canvas.height / 2;
      
      let action: number;
      if (Math.abs(ballY - paddleY) < 20) {
        action = 1; // Stay if close
      } else {
        action = ballY > paddleY ? 2 : 0; // Move toward ball
      }
      
      agent.lastAction = action;
      agent.actionCache.set(stateKey, action);
      return action;
    }

    // Batch prediction for efficiency
    const stateTensor = tf.tensor2d([state]);
    const qValues = agent.model.predict(stateTensor, { batchSize: 1 }) as tf.Tensor;
    const qArray = await qValues.data();
    stateTensor.dispose();
    qValues.dispose();

    const action = qArray.indexOf(Math.max(...Array.from(qArray)));
    agent.lastAction = action;
    
    // Cache the action
    agent.actionCache.set(stateKey, action);
    
    // Limit cache size
    if (agent.actionCache.size > 1000) {
      const keys = agent.actionCache.keys();
      const firstKey = keys.next().value;
      if (firstKey !== undefined) {
        agent.actionCache.delete(firstKey);
      }
    }

    return action;
  }

  // Enhanced reward system for faster learning
  private calculateEnhancedRewards(
    gameResult: { gameEnded: boolean, winner: number | null }, 
    prevState: number[], 
    newState: number[]
  ) {
    let agent1Reward = 0;
    let agent2Reward = 0;

    if (gameResult.gameEnded) {
      if (gameResult.winner === 1) {
        agent1Reward = 100; // Increased win reward
        agent2Reward = -100;
      } else if (gameResult.winner === 2) {
        agent1Reward = -100;
        agent2Reward = 100;
      }
    } else {
      // Enhanced shaping rewards for faster learning
      const ballX = (newState[0] + 1) * this.canvas.width / 2;
      const ballY = (newState[1] + 1) * this.canvas.height / 2;
      const ballDx = newState[2] * 10;
      
      // Paddle positioning rewards
      const paddle1Y = (newState[4] + 1) * this.canvas.height / 2;
      const paddle2Y = (newState[5] + 1) * this.canvas.height / 2;
      
      // Reward for keeping paddle aligned with ball
      const paddle1Distance = Math.abs(ballY - paddle1Y);
      const paddle2Distance = Math.abs(ballY - paddle2Y);
      
      agent1Reward = 2.0 - (paddle1Distance / 100); // Closer = better
      agent2Reward = 2.0 - (paddle2Distance / 100);
      
      // Extra reward for intercepting ball trajectory
      if (ballDx < 0 && ballX < this.canvas.width / 2) { // Ball moving to agent1
        agent1Reward += Math.max(0, 3.0 - paddle1Distance / 20);
      }
      if (ballDx > 0 && ballX > this.canvas.width / 2) { // Ball moving to agent2
        agent2Reward += Math.max(0, 3.0 - paddle2Distance / 20);
      }
      
      // Small penalty for staying idle
      if (this.agent1?.lastAction === 1) agent1Reward -= 0.1;
      if (this.agent2?.lastAction === 1) agent2Reward -= 0.1;
    }

    return { agent1: agent1Reward, agent2: agent2Reward };
  }

  private executeAction(player: number, action: number) {
    const paddle = player === 1 ? this.paddle1 : this.paddle2;
    const speed = 12; // Increased from 8 for faster movement

    switch (action) {
      case 0: // Move up
        paddle.y = Math.max(0, paddle.y - speed);
        break;
      case 1: // Stay
        break;
      case 2: // Move down
        paddle.y = Math.min(this.canvas.height - paddle.height, paddle.y + speed);
        break;
    }
  }

  private updateBall() {
    this.ball.x += this.ball.dx * this.config.gameSpeed;
    this.ball.y += this.ball.dy * this.config.gameSpeed;

    // Ball collision with top and bottom walls
    if (this.ball.y - this.ball.radius <= 0 || this.ball.y + this.ball.radius >= this.canvas.height) {
      this.ball.dy = -this.ball.dy;
    }

    // Ball collision with paddles
    if (this.ballCollidesWithPaddle(this.paddle1) || this.ballCollidesWithPaddle(this.paddle2)) {
      this.ball.dx = -this.ball.dx;
      // Add some randomness to make it more interesting
      this.ball.dy += (Math.random() - 0.5) * 2;
    }
  }

  private ballCollidesWithPaddle(paddle: any): boolean {
    return this.ball.x - this.ball.radius <= paddle.x + paddle.width &&
           this.ball.x + this.ball.radius >= paddle.x &&
           this.ball.y - this.ball.radius <= paddle.y + paddle.height &&
           this.ball.y + this.ball.radius >= paddle.y;
  }

  private checkGameEnd(): { gameEnded: boolean, winner: number | null } {
    if (this.ball.x < 0) {
      return { gameEnded: true, winner: 2 }; // Agent 2 wins
    } else if (this.ball.x > this.canvas.width) {
      return { gameEnded: true, winner: 1 }; // Agent 1 wins
    }
    return { gameEnded: false, winner: null };
  }

  private sampleBatch(memory: Experience[], batchSize: number): Experience[] {
    const batch: Experience[] = [];
    for (let i = 0; i < batchSize; i++) {
      const randomIndex = Math.floor(Math.random() * memory.length);
      batch.push(memory[randomIndex]);
    }
    return batch;
  }

  private updateTargetNetworks() {
    if (this.agent1 && this.agent2) {
      this.agent1.targetModel.setWeights(this.agent1.model.getWeights());
      this.agent2.targetModel.setWeights(this.agent2.model.getWeights());
    }
  }

  // Enhanced game end handling with logging
  private handleGameEnd(winner: number | null) {
    if (!this.agent1 || !this.agent2) return;

    this.agent1.gamesPlayed++;
    this.agent2.gamesPlayed++;

    let logMessage = '';
    if (winner === 1) {
      this.agent1.score++;
      this.stats.agent1Wins++;
      logMessage = `🔴 Red AI wins game ${this.gameNumber + 1}`;
    } else if (winner === 2) {
      this.agent2.score++;
      this.stats.agent2Wins++;
      logMessage = `🔵 Blue AI wins game ${this.gameNumber + 1}`;
    } else {
      logMessage = `⚽ Game ${this.gameNumber + 1} ended in a draw`;
    }

    this.stats.totalGames++;
    this.updateStats();

    // Log game results periodically
    if (this.gameNumber % 25 === 0) {
      this.addTrainingLog(logMessage);
    }

    // Log epsilon decay milestones
    if (this.agent1.epsilon <= 0.5 && this.agent1.epsilon > 0.49) {
      this.addTrainingLog('🎯 Exploration reduced to 50% - agents becoming more strategic');
    }
    if (this.agent1.epsilon <= 0.1 && this.agent1.epsilon > 0.09) {
      this.addTrainingLog('🧠 Exploration at 10% - agents mostly exploiting learned strategies');
    }
  }

  updateEpsilonDecay(value: number) {
    this.config.epsilonDecay = value;
    this.addTrainingLog(`Epsilon decay updated to ${value.toFixed(4)}`);
  }

  private updateVisualization() {
    // Update reward history
    if (this.agent1 && this.agent2) {
      this.rewardHistory.agent1.push(this.agent1.totalReward);
      this.rewardHistory.agent2.push(this.agent2.totalReward);
      
      this.winRateHistory.agent1.push(this.stats.agent1WinRate);
      this.winRateHistory.agent2.push(this.stats.agent2WinRate);
      
      // Keep only last 100 data points
      if (this.rewardHistory.agent1.length > 100) {
        this.rewardHistory.agent1.shift();
        this.rewardHistory.agent2.shift();
        this.winRateHistory.agent1.shift();
        this.winRateHistory.agent2.shift();
      }
    }
  }

  private render() {
    // Clear canvas
    this.ctx.fillStyle = '#0f0f23';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

    // Draw center line
    this.ctx.strokeStyle = '#00ff00';
    this.ctx.setLineDash([5, 5]);
    this.ctx.beginPath();
    this.ctx.moveTo(this.canvas.width / 2, 0);
    this.ctx.lineTo(this.canvas.width / 2, this.canvas.height);
    this.ctx.stroke();
    this.ctx.setLineDash([]);

    // Draw paddles with AI agent colors
    if (this.agent1) {
      this.ctx.fillStyle = this.agent1.color;
      this.ctx.fillRect(this.paddle1.x, this.paddle1.y, this.paddle1.width, this.paddle1.height);
    }

    if (this.agent2) {
      this.ctx.fillStyle = this.agent2.color;
      this.ctx.fillRect(this.paddle2.x, this.paddle2.y, this.paddle2.width, this.paddle2.height);
    }

    // Draw ball
    this.ctx.fillStyle = '#00ff00';
    this.ctx.beginPath();
    this.ctx.arc(this.ball.x, this.ball.y, this.ball.radius, 0, Math.PI * 2);
    this.ctx.fill();

    // Update neural network visualization
    this.updateNetworkVisualization();
  }

  private initializeNetworkVisualization() {
    if (!this.networkVisRef?.nativeElement) return;

    const container = d3.select(this.networkVisRef.nativeElement);
    container.selectAll('*').remove();

    // Create two network visualizations side by side
    const containerWidth = 1000;
    const containerHeight = 400;
    const networkWidth = 450;
    const networkHeight = 350;

    // Main container
    const mainSvg = container.append('svg')
      .attr('width', containerWidth)
      .attr('height', containerHeight)
      .style('background', 'rgba(15, 15, 35, 0.8)')
      .style('border', '2px solid #00ff00')
      .style('border-radius', '15px');

    // Agent 1 Network (Left side)
    const agent1Group = mainSvg.append('g')
      .attr('transform', 'translate(25, 25)');

    this.createAgentNetwork(agent1Group, networkWidth, networkHeight, 'agent1', '#ff4444');

    // Agent 2 Network (Right side)
    const agent2Group = mainSvg.append('g')
      .attr('transform', `translate(${25 + networkWidth + 50}, 25)`);

    this.createAgentNetwork(agent2Group, networkWidth, networkHeight, 'agent2', '#4444ff');

    // Add title
    mainSvg.append('text')
      .attr('x', containerWidth / 2)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .style('fill', '#00ff00')
      .style('font-family', 'monospace')
      .style('font-size', '16px')
      .style('font-weight', 'bold')
      .text('🧠 REAL-TIME NEURAL NETWORK VISUALIZATION');
  }

  private createAgentNetwork(container: any, width: number, height: number, agentId: string, color: string) {
    const layers = [8, 32, 16, 3]; // Simplified for better visualization
    const nodeRadius = 6;
    const layerSpacing = width / (layers.length + 1);
    
    // Agent title
    container.append('text')
      .attr('x', width / 2)
      .attr('y', 15)
      .attr('text-anchor', 'middle')
      .style('fill', color)
      .style('font-family', 'monospace')
      .style('font-size', '14px')
      .style('font-weight', 'bold')
      .text(`${agentId === 'agent1' ? 'RED AI' : 'BLUE AI'} BRAIN`);

    // Create nodes and links
    const nodes: any[] = [];
    const links: any[] = [];

    // Generate nodes
    for (let layerIndex = 0; layerIndex < layers.length; layerIndex++) {
      const layerSize = layerIndex === 1 ? 8 : layers[layerIndex]; // Limit middle layers for visualization
      const x = (layerIndex + 1) * layerSpacing;
      const startY = (height - (layerSize - 1) * 25) / 2;

      for (let nodeIndex = 0; nodeIndex < layerSize; nodeIndex++) {
        const y = startY + nodeIndex * 25;
        
        nodes.push({
          id: `${agentId}-${layerIndex}-${nodeIndex}`,
          x: x,
          y: y,
          layer: layerIndex,
          index: nodeIndex,
          agentId: agentId,
          activation: 0
        });

        // Create links to next layer
        if (layerIndex < layers.length - 1) {
          const nextLayerSize = layerIndex + 1 === 1 ? 8 : layers[layerIndex + 1];
          for (let nextNodeIndex = 0; nextNodeIndex < nextLayerSize; nextNodeIndex++) {
            links.push({
              id: `${agentId}-link-${layerIndex}-${nodeIndex}-${nextNodeIndex}`,
              source: `${agentId}-${layerIndex}-${nodeIndex}`,
              target: `${agentId}-${layerIndex + 1}-${nextNodeIndex}`,
              weight: (Math.random() - 0.5) * 2,
              agentId: agentId
            });
          }
        }
      }
    }

    // Draw connections first (so they appear behind nodes)
    container.selectAll('.network-link')
      .data(links)
      .enter()
      .append('line')
      .attr('class', `network-link ${agentId}-link`)
      .attr('x1', (d: any) => nodes.find(n => n.id === d.source)?.x ?? 0)
      .attr('y1', (d: any) => nodes.find(n => n.id === d.source)?.y ?? 0)
      .attr('x2', (d: any) => nodes.find(n => n.id === d.target)?.x ?? 0)
      .attr('y2', (d: any) => nodes.find(n => n.id === d.target)?.y ?? 0)
      .attr('stroke', '#666')
      .attr('stroke-width', 1)
      .attr('stroke-opacity', 0.2);

    // Draw nodes
    container.selectAll('.network-node')
      .data(nodes)
      .enter()
      .append('circle')
      .attr('class', `network-node ${agentId}-node`)
      .attr('cx', (d: any) => d.x)
      .attr('cy', (d: any) => d.y)
      .attr('r', nodeRadius)
      .attr('fill', '#333')
      .attr('stroke', color)
      .attr('stroke-width', 1);

    // Add input labels
    const inputLabels = ['Ball X', 'Ball Y', 'Ball ΔX', 'Ball ΔY', 'My Paddle', 'Enemy Paddle', 'Distance X', 'Distance Y'];
    nodes.filter(n => n.layer === 0).forEach((node, i) => {
      container.append('text')
        .attr('x', node.x - 45)
        .attr('y', node.y + 3)
        .attr('text-anchor', 'end')
        .style('fill', '#ccc')
        .style('font-family', 'monospace')
        .style('font-size', '8px')
        .text(inputLabels[i] || `Input ${i + 1}`);
    });

    // Add output labels
    const outputLabels = ['Move Up', 'Stay', 'Move Down'];
    nodes.filter(n => n.layer === layers.length - 1).forEach((node, i) => {
      container.append('text')
        .attr('x', node.x + 15)
        .attr('y', node.y + 3)
        .attr('text-anchor', 'start')
        .style('fill', '#ccc')
        .style('font-family', 'monospace')
        .style('font-size', '8px')
        .text(outputLabels[i] || `Output ${i + 1}`);
    });

    // Add layer labels
    const layerLabels = ['INPUTS', 'HIDDEN', 'HIDDEN', 'ACTIONS'];
    for (let i = 0; i < layers.length; i++) {
      container.append('text')
        .attr('x', (i + 1) * layerSpacing)
        .attr('y', height - 10)
        .attr('text-anchor', 'middle')
        .style('fill', color)
        .style('font-family', 'monospace')
        .style('font-size', '10px')
        .style('font-weight', 'bold')
        .text(layerLabels[i]);
    }

    // Add activation indicator
    container.append('text')
      .attr('class', `${agentId}-activation-text`)
      .attr('x', width / 2)
      .attr('y', height + 15)
      .attr('text-anchor', 'middle')
      .style('fill', color)
      .style('font-family', 'monospace')
      .style('font-size', '10px')
      .text('Thinking...');
  }

  private async updateNetworkVisualization() {
    if (!this.networkVisRef?.nativeElement || !this.agent1 || !this.agent2) return;

    const state = this.getGameState();

    // Get actual predictions from both agents if they exist
    let agent1Predictions: number[] = [];
    let agent2Predictions: number[] = [];

    try {
      if (this.agent1) {
        const stateTensor = tf.tensor2d([state]);
        const predictions = this.agent1.model.predict(stateTensor) as tf.Tensor;
        agent1Predictions = Array.from(await predictions.data());
        stateTensor.dispose();
        predictions.dispose();
      }

      if (this.agent2) {
        const stateTensor = tf.tensor2d([state]);
        const predictions = this.agent2.model.predict(stateTensor) as tf.Tensor;
        agent2Predictions = Array.from(await predictions.data());
        stateTensor.dispose();
        predictions.dispose();
      }
    } catch (error) {
      console.warn('Error getting network predictions:', error);
    }

    // Update Agent 1 network
    this.updateAgentVisualization('agent1', state, agent1Predictions, '#ff4444');
    
    // Update Agent 2 network  
    this.updateAgentVisualization('agent2', state, agent2Predictions, '#4444ff');
  }

  private updateAgentVisualization(agentId: string, state: number[], predictions: number[], color: string) {
    const svg = d3.select(this.networkVisRef.nativeElement).select('svg');

    // Update input layer with actual game state
    svg.selectAll(`.${agentId}-node`)
      .filter((d: any) => d.layer === 0)
      .transition()
      .duration(100)
      .attr('fill', (d: any) => {
        const activation = Math.abs(state[d.index] || 0);
        const intensity = Math.min(activation, 1);
        return d3.interpolateRgb('#000', color)(intensity * 0.8 + 0.2);
      })
      .attr('r', (d: any) => {
        const activation = Math.abs(state[d.index] || 0);
        return 6 + activation * 3; // Pulse based on activation
      });

    // Update hidden layers with simulated activations based on input
    svg.selectAll(`.${agentId}-node`)
      .filter((d: any) => d.layer > 0 && d.layer < 3)
      .transition()
      .duration(150)
      .attr('fill', (d: any) => {
        // Simulate realistic hidden layer activations
        const baseActivation = state.reduce((sum, val, i) => sum + Math.abs(val) * Math.sin(i + d.index), 0) / state.length;
        const activation = Math.abs(Math.sin(baseActivation + Date.now() / 1000 + d.index)) * 0.7 + 0.3;
        return d3.interpolateRgb('#000', color)(activation);
      })
      .attr('r', 6);

    // Update output layer with actual predictions
    if (predictions.length >= 3) {
      const maxPrediction = Math.max(...predictions);
      const chosenAction = predictions.indexOf(maxPrediction);

      svg.selectAll(`.${agentId}-node`)
        .filter((d: any) => d.layer === 3)
        .transition()
        .duration(100)
        .attr('fill', (d: any) => {
          const prediction = predictions[d.index] || 0;
          const normalized = (prediction + 10) / 20; // Normalize Q-values roughly to 0-1
          const intensity = Math.max(0, Math.min(1, normalized));
          return d3.interpolateRgb('#000', color)(intensity);
        })
        .attr('r', (d: any) => {
          const isChosen = d.index === chosenAction;
          return isChosen ? 10 : 6; // Highlight chosen action
        })
        .attr('stroke-width', (d: any) => {
          const isChosen = d.index === chosenAction;
          return isChosen ? 3 : 1;
        });

      // Update action text
      const actionNames = ['Moving Up', 'Staying', 'Moving Down'];
      svg.select(`.${agentId}-activation-text`)
        .text(`Decision: ${actionNames[chosenAction]} (Q: ${maxPrediction.toFixed(2)})`);
    }

    // Update connection weights to show information flow
    svg.selectAll(`.${agentId}-link`)
      .transition()
      .duration(200)
      .attr('stroke-opacity', () => 0.1 + Math.random() * 0.3)
      .attr('stroke-width', () => 0.5 + Math.random() * 1.5)
      .attr('stroke', () => {
        const intensity = Math.random();
        return intensity > 0.7 ? color : '#666';
      });
  }

  getAgent1Stats() {
    if (!this.agent1) return null;
    
    return {
      name: 'RED AI',
      wins: this.stats.agent1Wins,
      winRate: this.stats.totalGames > 0 ? ((this.stats.agent1Wins / this.stats.totalGames) * 100).toFixed(1) : '0.0',
      avgReward: this.stats.agent1Rewards.length > 0 ? 
        (this.stats.agent1Rewards.reduce((a, b) => a + b, 0) / this.stats.agent1Rewards.length).toFixed(2) : '0.00',
      epsilon: this.agent1.epsilon.toFixed(3),
      gamesPlayed: this.stats.totalGames
    };
  }

  getAgent2Stats() {
    if (!this.agent2) return null;
    
    return {
      name: 'BLUE AI',
      wins: this.stats.agent2Wins,
      winRate: this.stats.totalGames > 0 ? ((this.stats.agent2Wins / this.stats.totalGames) * 100).toFixed(1) : '0.0',
      avgReward: this.stats.agent2Rewards.length > 0 ? 
        (this.stats.agent2Rewards.reduce((a, b) => a + b, 0) / this.stats.agent2Rewards.length).toFixed(2) : '0.00',
      epsilon: this.agent2.epsilon.toFixed(3),
      gamesPlayed: this.stats.totalGames
    };
  }

  getFPS(): number {
    return this.fps;
  }

  getTrainingSpeed(): string {
    return `${this.config.stepsPerFrame * this.config.gameSpeed}x`;
  }

  private getGameState(): number[] {
    // Normalize all values to [-1, 1] range
    return [
      (this.ball.x / this.canvas.width) * 2 - 1,
      (this.ball.y / this.canvas.height) * 2 - 1,
      this.ball.dx / 10,
      this.ball.dy / 10,
      (this.paddle1.y / this.canvas.height) * 2 - 1,
      (this.paddle2.y / this.canvas.height) * 2 - 1,
      (this.ball.x - this.paddle1.x) / this.canvas.width,
      (this.ball.x - this.paddle2.x) / this.canvas.width
    ];
  }

  // Batched training for better performance
  private async processBatchedTraining() {
    if (this.trainingBuffer.length === 0) return;

    // Group by agent
    const agent1Experiences = this.trainingBuffer.filter(exp => exp.agent.id === 'agent1');
    const agent2Experiences = this.trainingBuffer.filter(exp => exp.agent.id === 'agent2');

    // Train both agents in parallel
    const trainingPromises = [];
    
    if (agent1Experiences.length >= this.config.batchSize / 2) {
      trainingPromises.push(this.trainAgentBatched(this.agent1!, agent1Experiences));
    }
    
    if (agent2Experiences.length >= this.config.batchSize / 2) {
      trainingPromises.push(this.trainAgentBatched(this.agent2!, agent2Experiences));
    }

    await Promise.all(trainingPromises);
    
    // Clear buffer
    this.trainingBuffer.length = 0;
  }

  // Optimized training with Double DQN and prioritized replay
  private async trainAgentBatched(agent: AIAgent, experiences: any[]) {
    if (experiences.length < 8) return; // Minimum batch size

    // Add experiences to memory
    experiences.forEach(exp => {
      agent.memory.push({
        state: exp.state,
        action: exp.action,
        reward: exp.reward,
        nextState: exp.nextState,
        done: exp.done
      });
      agent.totalReward += exp.reward;
    });

    // Limit memory size
    while (agent.memory.length > this.config.memorySize) {
      agent.memory.shift();
    }

    if (agent.memory.length < this.config.batchSize) return;

    // Sample batch with prioritized replay
    const batch = this.config.prioritizedReplay ? 
      this.samplePrioritizedBatch(agent.memory, Math.min(this.config.batchSize, experiences.length * 2)) :
      this.sampleBatch(agent.memory, Math.min(this.config.batchSize, experiences.length * 2));
    
    const states = batch.map(exp => exp.state);
    const nextStates = batch.map(exp => exp.nextState);

    // Get predictions in batches
    const currentQs = agent.model.predict(tf.tensor2d(states), { batchSize: batch.length }) as tf.Tensor;
    const nextQs = agent.targetModel.predict(tf.tensor2d(nextStates), { batchSize: batch.length }) as tf.Tensor;
    
    let nextQsMain: tf.Tensor | null = null;
    if (this.config.doubleQLearning) {
      nextQsMain = agent.model.predict(tf.tensor2d(nextStates), { batchSize: batch.length }) as tf.Tensor;
    }
    
    const currentQsArray = await currentQs.data();
    const nextQsArray = await nextQs.data();
    const nextQsMainArray = nextQsMain ? await nextQsMain.data() : null;

    // Prepare training data with Double DQN
    const trainX: number[][] = [];
    const trainY: number[][] = [];

    for (let i = 0; i < batch.length; i++) {
      const exp = batch[i];
      const currentQ = Array.from(currentQsArray.slice(i * 3, (i + 1) * 3));
      const nextQ = Array.from(nextQsArray.slice(i * 3, (i + 1) * 3));
      
      let targetQ = exp.reward;
      if (!exp.done) {
        if (this.config.doubleQLearning && nextQsMainArray) {
          // Double DQN: use main network to select action, target network to evaluate
          const nextQMain = Array.from(nextQsMainArray.slice(i * 3, (i + 1) * 3));
          const bestAction = nextQMain.indexOf(Math.max(...nextQMain));
          targetQ = exp.reward + 0.99 * nextQ[bestAction]; // Higher gamma for better long-term planning
        } else {
          targetQ = exp.reward + 0.99 * Math.max(...nextQ);
        }
      }
      
      currentQ[exp.action] = targetQ;
      trainX.push(exp.state);
      trainY.push(currentQ);
    }

    // Train with optimized settings
    const xs = tf.tensor2d(trainX);
    const ys = tf.tensor2d(trainY);
    
    await agent.model.fit(xs, ys, { 
      epochs: 1, 
      verbose: 0,
      batchSize: Math.min(32, batch.length),
      shuffle: true
    });

    // Cleanup tensors
    currentQs.dispose();
    nextQs.dispose();
    if (nextQsMain) nextQsMain.dispose();
    xs.dispose();
    ys.dispose();

    // Faster epsilon decay
    agent.epsilon = Math.max(this.config.epsilonMin, agent.epsilon * this.config.epsilonDecay);
  }

  // Prioritized experience replay
  private samplePrioritizedBatch(memory: Experience[], batchSize: number): Experience[] {
    // Simple prioritized sampling - prefer recent experiences and high rewards
    const recentExperiences = memory.slice(-Math.floor(memory.length * 0.3));
    const highRewardExperiences = memory.filter(exp => Math.abs(exp.reward) > 1.0);
    
    const prioritizedPool = [...recentExperiences, ...highRewardExperiences];
    
    const batch: Experience[] = [];
    for (let i = 0; i < Math.min(batchSize, prioritizedPool.length); i++) {
      const randomIndex = Math.floor(Math.random() * prioritizedPool.length);
      batch.push(prioritizedPool[randomIndex]);
    }
    
    // Fill remaining with random samples if needed
    while (batch.length < batchSize && batch.length < memory.length) {
      const randomIndex = Math.floor(Math.random() * memory.length);
      batch.push(memory[randomIndex]);
    }
    
    return batch;
  }

  // FPS tracking
  private updateFPS(startTime: number) {
    const currentTime = performance.now();
    const deltaTime = currentTime - this.lastFrameTime;
    
    if (deltaTime > 0) {
      this.fps = Math.round(1000 / deltaTime);
    }
    
    this.lastFrameTime = currentTime;
  }

  private resetGame() {
    this.ball.x = this.canvas.width / 2;
    this.ball.y = this.canvas.height / 2;
    this.ball.dx = (Math.random() > 0.5 ? 1 : -1) * 8; // Faster initial speed
    this.ball.dy = (Math.random() - 0.5) * 8;
    
    this.paddle1.y = (this.canvas.height - this.paddle1.height) / 2;
    this.paddle2.y = (this.canvas.height - this.paddle2.height) / 2;
    
    // Clear action caches for new game
    if (this.agent1) this.agent1.actionCache.clear();
    if (this.agent2) this.agent2.actionCache.clear();
  }

  // Enhanced configuration update methods
  updateLearningRate(value: number) {
    this.config.learningRate = value;
    this.addTrainingLog(`Learning rate updated to ${value.toFixed(4)}`);
    if (this.agent1 && this.agent2) {
      // Recompile models with new learning rate
      const optimizer = tf.train.adam(this.config.learningRate, 0.9, 0.999, 1e-7);
      this.agent1.model.compile({ optimizer, loss: 'huberLoss', metrics: ['mse'] });
      this.agent2.model.compile({ optimizer, loss: 'huberLoss', metrics: ['mse'] });
    }
  }

  updateGameSpeed(value: number) {
    this.config.gameSpeed = value;
    this.addTrainingLog(`Game speed updated to ${value}x`);
  }

  updateStepsPerFrame(value: number) {
    this.config.stepsPerFrame = value;
    this.addTrainingLog(`Steps per frame updated to ${value}`);
  }

  // Add training log entry
  addTrainingLog(message: string) {
    const log: TrainingLog = {
      timestamp: new Date().toLocaleTimeString(),
      gameNumber: this.gameNumber,
      agent1Score: this.agent1?.score ?? 0,
      agent2Score: this.agent2?.score ?? 0,
      agent1Reward: this.agent1?.totalReward ?? 0,
      agent2Reward: this.agent2?.totalReward ?? 0,
      agent1Epsilon: this.agent1?.epsilon ?? 0,
      agent2Epsilon: this.agent2?.epsilon ?? 0,
      message: message
    };

    this.trainingLogs.unshift(log);
    
    // Keep only the last maxLogs entries
    if (this.trainingLogs.length > this.maxLogs) {
      this.trainingLogs = this.trainingLogs.slice(0, this.maxLogs);
    }
  }

  private updateStats() {
    if (this.stats.totalGames > 0) {
      this.stats.agent1WinRate = (this.stats.agent1Wins / this.stats.totalGames) * 100;
      this.stats.agent2WinRate = (this.stats.agent2Wins / this.stats.totalGames) * 100;
    }

    if (this.agent1 && this.agent2) {
      this.stats.avgRewardAgent1 = this.agent1.totalReward / Math.max(1, this.agent1.gamesPlayed);
      this.stats.avgRewardAgent2 = this.agent2.totalReward / Math.max(1, this.agent2.gamesPlayed);
    }
  }

  // Export/Import model functionality
  async exportModel(agentType: 'agent1' | 'agent2') {
    const agent = agentType === 'agent1' ? this.agent1 : this.agent2;
    if (!agent) return;

    try {
      // Create a downloadable model file
      const modelData = {
        modelWeights: agent.model.getWeights().map(w => w.arraySync()),
        config: {
          epsilon: agent.epsilon,
          totalReward: agent.totalReward,
          gamesPlayed: agent.gamesPlayed,
          score: agent.score
        },
        exportDate: new Date().toISOString(),
        agentType: agentType
      };

      const blob = new Blob([JSON.stringify(modelData, null, 2)], {
        type: 'application/json'
      });
      
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `pong-ai-${agentType}-${Date.now()}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      this.addTrainingLog(`${agentType} model exported successfully`);
    } catch (error) {
      console.error('Error exporting model:', error);
      this.addTrainingLog(`Error exporting ${agentType} model`);
    }
  }

  async importModel(agentType: 'agent1' | 'agent2', file: File) {
    if (!file) return;

    try {
      this.addTrainingLog(`📥 Importing model for ${agentType}...`);
      
      const text = await file.text();
      const modelData = JSON.parse(text);

      const agent = agentType === 'agent1' ? this.agent1 : this.agent2;
      if (!agent) {
        this.addTrainingLog(`❌ Cannot import to ${agentType} - agent not initialized`);
        return;
      }

      // Restore weights
      const weights = modelData.modelWeights.map((w: any) => tf.tensor(w));
      agent.model.setWeights(weights);
      agent.targetModel.setWeights(weights);

      // Restore stats if available
      if (modelData.config) {
        agent.epsilon = modelData.config.epsilon ?? agent.epsilon;
        agent.totalReward = modelData.config.totalReward ?? 0;
        agent.gamesPlayed = modelData.config.gamesPlayed ?? 0;
        agent.score = modelData.config.score ?? 0;
      }

      // Cleanup tensors
      weights.forEach((w: tf.Tensor) => w.dispose());

      this.addTrainingLog(`✅ Model imported successfully for ${agent.name}`);
      this.addTrainingLog(`📊 Restored stats: ${agent.gamesPlayed} games, ${agent.score} wins`);
    } catch (error) {
      console.error('Error importing model:', error);
      this.addTrainingLog(`❌ Failed to import model for ${agentType}`);
    }
  }

  // Reset training progress
  resetTraining() {
    this.stopTraining();
    
    this.addTrainingLog('🔄 Resetting training session...');
    
    // Reset statistics
    this.stats = {
      agent1Wins: 0,
      agent2Wins: 0,
      totalGames: 0,
      avgRewardAgent1: 0,
      avgRewardAgent2: 0,
      agent1WinRate: 0,
      agent2WinRate: 0,
      agent1Rewards: [] as number[],
      agent2Rewards: [] as number[]
    };

    // Clear history
    this.rewardHistory.agent1.length = 0;
    this.rewardHistory.agent2.length = 0;
    this.winRateHistory.agent1.length = 0;
    this.winRateHistory.agent2.length = 0;
    this.lossHistory.length = 0;

    // Reset game counter
    this.gameNumber = 0;
    this.episode = 0;

    // Dispose and recreate agents if they exist
    if (this.agent1) {
      this.agent1.model.dispose();
      this.agent1.targetModel.dispose();
      this.agent1 = null;
    }
    
    if (this.agent2) {
      this.agent2.model.dispose();
      this.agent2.targetModel.dispose();
      this.agent2 = null;
    }

    // Clear visualization
    if (this.networkVisRef?.nativeElement) {
      d3.select(this.networkVisRef.nativeElement).selectAll('*').remove();
    }

    this.addTrainingLog('✅ Training session reset complete');
    this.addTrainingLog('🎯 Ready to start new training session');
  }

  // Navigate back to menu
  goBackToMenu() {
    this.stopTraining();
    this.backToMenu.emit();
  }

  // TrackBy function for ngFor performance
  trackByIndex(index: number, item: any): number {
    return index;
  }

  // Clean up resources
  private cleanup() {
    if (this.agent1) {
      this.agent1.model.dispose();
      this.agent1.targetModel.dispose();
    }
    
    if (this.agent2) {
      this.agent2.model.dispose();
      this.agent2.targetModel.dispose();
    }

    // Clear D3 visualization
    if (this.networkVisRef?.nativeElement) {
      d3.select(this.networkVisRef.nativeElement).selectAll('*').remove();
    }
  }
}